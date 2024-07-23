import sys
import math
from typing import Iterable

import torch
from torch import nn
import numpy as np

import datasets as myDBs
from torchvision import transforms as vT
from util import audio_transforms as aT

from models.deepavfusion import DeepAVFusion
from models.avsrcsep import AVSrcSep
from mir_eval.separation import bss_eval_sources

from util import distributed as dist_utils
from util import misc as misc_utils
from util import data as data_utils
from util import meters, lr_sched


def main_worker(local_rank, args):
    # Setup environment
    job_dir = f"{args.output_dir}/{args.job_name}"
    dist_utils.init_distributed_mode(local_rank, args, log_fn=f"{job_dir}/train.log")
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    print(f'job dir: {job_dir}')
    misc_utils.print_args(args)

    # Adjust learning rate to batch size
    num_tasks = dist_utils.get_world_size()
    num_tasks_per_node = max(1, torch.cuda.device_count())
    args.env.workers = args.env.workers // num_tasks_per_node
    eff_batch_size = args.opt.batch_size * args.opt.accum_iter * num_tasks
    if args.opt.lr is None:  # only base_lr is specified
        args.opt.lr = args.opt.blr * eff_batch_size / 256
    print("base lr: %.2e" % args.opt.blr)
    print("actual lr: %.2e" % args.opt.lr)
    print("accumulate grad iterations: %d" % args.opt.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Dataloaders
    dataset_train = myDBs.load_dataset(
        args.data.dataset,
        args.data.data_path,
        dataset_type='mixed_audio',
        visual_transform=vT.Compose([
            vT.RandomResizedCrop(args.data.image_size, scale=(args.data.crop_min, 1.)),
            vT.RandomHorizontalFlip(),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        audio_transform=aT.Compose([
            aT.Pad(rate=args.data.audio_rate, dur=args.data.audio_dur),
            aT.MelSpectrogram(sample_rate=args.data.audio_rate, n_fft=int(args.data.audio_rate * 0.05), hop_length=int(args.data.audio_rate / 64), n_mels=args.data.audio_mels),
            aT.Log()
        ]),
        train=True,
        num_mixtures=args.avss.num_mixtures,
        audio_dur=args.data.audio_dur,
        audio_rate=args.data.audio_rate,
        temporal_jitter=True
    )
    loader_train = data_utils.get_dataloader(
        dataset_train, args.env.distributed, args.opt.batch_size, args.env.workers, shuffle=True, drop_last=True)
    print(dataset_train)

    dataset_val = myDBs.load_dataset(
        args.data.dataset,
        args.data.data_path,
        dataset_type='mixed_audio',
        visual_transform=vT.Compose([
            vT.Resize(int(args.data.image_size/0.875)),
            vT.CenterCrop(args.data.image_size),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        audio_transform=aT.Compose([
            aT.Pad(rate=args.data.audio_rate, dur=args.data.audio_dur),
            aT.MelSpectrogram(sample_rate=args.data.audio_rate, n_fft=int(args.data.audio_rate * 0.05), hop_length=int(args.data.audio_rate / 64), n_mels=args.data.audio_mels),
            aT.Log()]),
        train=False,
        num_mixtures=args.avss.num_mixtures,
        audio_dur=args.data.audio_dur,
        audio_rate=args.data.audio_rate,
    )
    loader_val = data_utils.get_dataloader(
        dataset_val, args.env.distributed, args.opt.batch_size, args.env.workers, shuffle=False, drop_last=False)
    print(dataset_val)

    # Create model
    image_size, audio_size = (args.data.image_size, args.data.image_size), (args.data.audio_mels, int(args.data.audio_dur*64))
    encoder = DeepAVFusion(
        image_arch=args.model.image.backbone, image_pretrained=args.model.image.pretrained, image_size=image_size,
        audio_arch=args.model.audio.backbone, audio_pretrained=args.model.audio.pretrained, audio_size=audio_size,
        fusion_arch=args.model.fusion.arch,
        fusion_layers=args.model.fusion.layers,
        num_fusion_tkns=(args.model.fusion.num_fusion_tkns,
                         args.model.fusion.num_aggr_image_tkns,
                         args.model.fusion.num_aggr_audio_tkns),
        drop_path=args.opt.drop_path,
        attn_drop=args.opt.attn_drop,
        drop=args.opt.proj_drop,
        fusion_mlp_ratio=args.model.fusion.mlp_ratio,
        fusion_attn_ratio=args.model.fusion.attn_ratio,
        fusion_num_heads=args.model.fusion.num_heads
    )
    model = AVSrcSep(encoder=encoder,
                     log_freq=args.avss.log_freq,
                     weighted_loss=args.avss.weighted_loss,
                     binary_mask=args.avss.binary_mask)
    model.to(device)
    print("Model = %s" % str(model))

    if args.checkpoint or args.pretrain_job_name:
        pretrain_ckpt = args.checkpoint or f"{args.output_dir}/checkpoints/checkpoint_{args.pretrain_resume_epoch}.pth"
        encoder.load_checkpoint(pretrain_ckpt, args.encoder_prefix)

    # Optimizer with layer-wise lr decay (lrd)
    param_groups = lr_sched.param_groups_lrd(
        model, args.opt.weight_decay,
        no_weight_decay_list=[n for n, p in model.named_parameters() if 'bias' in n or 'norm' in n],
        layer_decay=args.opt.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.opt.lr)

    # Trainer
    trainer = misc_utils.Trainer(
        model,
        optimizer=optimizer,
        use_amp=args.opt.use_amp,
        accum_iter=args.opt.accum_iter,
        distributed=args.env.distributed
    )

    # Checkpointing and logging
    ckpt_manager = misc_utils.CheckpointManager(
        modules=trainer.module_dict(),
        ckpt_dir=f"{job_dir}/checkpoints",
        epochs=args.opt.epochs,
        save_freq=args.log.save_freq)
    start_epoch = ckpt_manager.resume()[0] if args.opt.resume else 0
    wb_logger = misc_utils.WBLogger(
        f"{job_dir}/wandb", args.log.wandb_entity, args.log.wandb_project+'-avsrcsep', args.job_name,
        model, args)

    if args.eval:
        evaluate(trainer.eval_model, loader_val, start_epoch, device, args)
        exit(0)

    # =============================================================== #
    # Training loop
    print(f"Start training for {args.opt.epochs} epochs")
    for epoch in range(start_epoch, args.opt.epochs):
        if args.env.distributed:
            loader_train.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(trainer, loader_train, epoch,
                        device=device, wb_logger=wb_logger, args=args)

        # evaluate
        if epoch % args.log.eval_freq == 0 or epoch == args.opt.epochs - 1 or epoch == start_epoch:
            global_step = (len(loader_train) // trainer.accum_iter) * (epoch + 1)
            test_stats = evaluate(trainer.eval_model, loader_val, epoch, device, args)
            wb_logger.log(test_stats, step=global_step, force=True)

        # save checkpoint
        ckpt_manager.checkpoint(epoch+1, {'epoch': epoch+1})


def train_one_epoch(trainer: misc_utils.Trainer,
                    loader: Iterable,
                    epoch: int = 0,
                    wb_logger: misc_utils.WBLogger = None,
                    device: torch.device = torch.device('cpu'),
                    args=None):
    trainer.model.train(True)
    metric_logger = meters.MetricLogger(delimiter="  ")
    header = f'[Train][Ep-{epoch}/{args.opt.epochs}]'

    trainer.zero_grad()
    for step, (image, audio_mix, anno) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        sys.stdout.flush()
        global_step = (len(loader) // trainer.accum_iter) * epoch + step // trainer.accum_iter
        if step % args.opt.accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(trainer.optimizer, epoch + step / len(loader), args)
            metric_logger.update(lr=lr)

        # Prepare data
        image = image[0].to(device, non_blocking=True).float()
        audio_mix = audio_mix.to(device, non_blocking=True).float()
        audio_trg = anno['mel_specs'][:, 0].to(device, non_blocking=True).float()

        # Forward pass
        with trainer.autocast(), trainer.autosync():
            loss = trainer.model(image, audio_mix, audio_trg)[0]

        if not math.isfinite(loss.item()):
            raise f"Loss is {loss.item()}, stopping training"

        # Backward pass and model update
        grad_norm, amp_scale = trainer.step(loss)

        # Log
        if trainer.accums == 0:
            metric_logger.update(loss=loss.item(), grad_norm=grad_norm, amp_scale=amp_scale, n=image.shape[0])
            wb_logger.log(metric_logger.latest(), step=global_step)

        if args.debug and step == 100:
            break

    # gather the stats from all processes
    print("Syncing meters...")
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    trainer.zero_grad()
    return metric_logger.averages()


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: Iterable,
             epoch: int = 0,
             device: torch.device = torch.device('cpu'),
             args=None):
    model.train(False)
    metric_logger = meters.MetricLogger(delimiter="  ")
    header = f'[Eval][Ep-{epoch}/{args.opt.epochs}]'

    evaluator = AVSrcSepEvaluator()
    masking_separation = SpectrogramMasking(args.data.audio_rate, args.data.audio_mels)
    for step, (image, audio, anno, name) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        # Prepare data
        audio = audio.to(device, non_blocking=True).float()
        audio_mix = anno['mixed_audio'].to(device, non_blocking=True).float()
        frames1 = anno['frames'][:, 0].to(device, non_blocking=True).float()
        frames2 = anno['frames'][:, 1].to(device, non_blocking=True).float()

        # Separate
        _, pred_mask1, _ = model(frames1, audio_mix, audio)
        _, pred_mask2, _ = model(frames2, audio_mix, audio)

        # Compute separation metrics
        mix_waveforms = anno['waveforms'].sum(1)
        for i in range(audio.shape[0]):
            waveform_gt = anno['waveforms'][i].squeeze(1)
            waveform_pred1 = masking_separation(mix_waveforms[i], pred_mask1[i])
            waveform_pred2 = masking_separation(mix_waveforms[i], pred_mask2[i])
            waveform_pred = np.stack((waveform_pred1, waveform_pred2), axis=0).squeeze(1)
            if torch.any(waveform_gt.pow(2).sum(-1) < 1e-5):
                continue
            if np.any((waveform_pred**2).sum(-1) < 1e-5):
                continue
            evaluator.update(waveform_gt, waveform_pred, name[i])

        if args.debug and step == 8:
            break

    sdr, sir, sar = evaluator.average_sdr_sir_sar()
    print(f'{header} SDR={round(sdr,5)} SIR={round(sir,5)}  SAR={round(sar,5)}')
    return {'sdr': sdr, 'sir': sir, 'sar': sar}


class SpectrogramMasking:
    def __init__(self, audio_rate, audio_mels):
        n_fft = int(audio_rate * 0.05)
        hop_length = int(audio_rate / 64)
        self.spectrogram_t = aT.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.inv_spectrogram_t = aT.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mel_spec = aT.MelSpectrogram(sample_rate=audio_rate, n_fft=n_fft, hop_length=hop_length, n_mels=audio_mels)

    def __call__(self, waveform_mix, pred_mask):
        stft_mix = self.spectrogram_t(waveform_mix)
        pred_mask = torch.cat((torch.sigmoid(pred_mask).detach().cpu(), torch.zeros(*pred_mask.shape[:2], 1)), dim=2)
        pred_mask = torch.einsum('bmt,fm->bft', pred_mask, self.mel_spec.mel_scale.fb)
        waveform_pred = self.inv_spectrogram_t(pred_mask * stft_mix)
        return waveform_pred


class AVSrcSepEvaluator(object):
    def __init__(self, ):
        super(AVSrcSepEvaluator, self).__init__()

        self.name_list = []
        self.sdr_list = []
        self.sir_list = []
        self.sar_list = []

    def average_sdr_sir_sar(self):
        sdr = np.mean(self.sdr_list)
        sir = np.mean(self.sir_list)
        sar = np.mean(self.sar_list)
        return sdr, sir, sar

    def clear(self):
        self.name_list = []
        self.sdr_list = []
        self.sir_list = []
        self.sar_list = []

    def update(self, waveform_gt, waveform_pred, name):
        if isinstance(waveform_gt, torch.Tensor):
            waveform_gt = waveform_gt.detach().cpu().numpy()
        if isinstance(waveform_pred, torch.Tensor):
            waveform_pred = waveform_pred.detach().cpu().numpy()

        sdr, sir, sar, _ = bss_eval_sources(waveform_gt, waveform_pred, False)

        # Save
        self.name_list.append(name)
        self.sdr_list.append(sdr)
        self.sir_list.append(sir)
        self.sar_list.append(sar)