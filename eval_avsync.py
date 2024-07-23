import sys
import math
from typing import Iterable

import torch
from torch import nn
import numpy as np

import datasets as myDBs
from torchvision import transforms as T
from pytorchvideo import transforms as vT
from util import audio_transforms as aT

from models.video_earlyfusion import VideoEarlyFusion
from models.classifier import AVClassifier

from util import distributed as dist_utils
from util import misc as misc_utils
from util import data as data_utils
from util import meters, lr_sched


def main_worker(local_rank, args):
    # Setup environment
    job_dir = f"{args.output_dir}/{args.job_name}"
    print(f'job dir: {job_dir}')
    dist_utils.init_distributed_mode(local_rank, args, log_fn=f"{job_dir}/train.log")
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
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
        dataset_type='avsync',
        visual_transform=T.Compose([
            vT.UniformTemporalSubsample(args.data.num_frames),
            vT.ConvertUint8ToFloat(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vT.RandomResizedCrop(args.data.crop_size, args.data.crop_size, scale=(args.data.crop_min, 1.), aspect_ratio=(3/4, 4/3)),
            T.RandomHorizontalFlip()]),
        audio_transform=T.Compose([
            aT.Pad(rate=args.data.audio_rate, dur=args.data.audio_dur),
            aT.MelSpectrogram(sample_rate=args.data.audio_rate, n_fft=int(args.data.audio_rate * 0.05), hop_length=int(args.data.audio_rate / 64), n_mels=args.data.audio_mels),
            aT.Log()]),
        train=True, temporal_jitter=True,
        audio_dur=args.data.audio_dur, audio_rate=args.data.audio_rate,
        video_dur=args.data.num_frames/args.data.video_rate, video_rate=args.data.video_rate,
    )
    loader_train = data_utils.get_dataloader(
        dataset_train, args.env.distributed, args.opt.batch_size, args.env.workers, shuffle=True, drop_last=True)
    print(dataset_train)

    dataset_val = myDBs.load_dataset(
        args.data.dataset,
        args.data.data_path,
        dataset_type='avsync',
        visual_transform=T.Compose([
            vT.UniformTemporalSubsample(args.data.num_frames),
            vT.ConvertUint8ToFloat(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vT.ShortSideScale(int(args.data.crop_size/0.875)),
            T.CenterCrop(args.data.crop_size)]),
        audio_transform=aT.Compose([
            aT.Pad(rate=args.data.audio_rate, dur=args.data.audio_dur),
            aT.MelSpectrogram(sample_rate=args.data.audio_rate, n_fft=int(args.data.audio_rate * 0.05), hop_length=int(args.data.audio_rate / 64), n_mels=args.data.audio_mels),
            aT.Log()]),
        train=False, temporal_jitter=False,
        audio_dur=args.data.audio_dur, audio_rate=args.data.audio_rate,
        video_dur=args.data.num_frames/args.data.video_rate, video_rate=args.data.video_rate,
    )
    loader_val = data_utils.get_dataloader(
        dataset_val, args.env.distributed, args.opt.batch_size, args.env.workers, shuffle=False, drop_last=False)
    print(dataset_val)

    # Create model
    video_size = (args.data.num_frames, args.data.crop_size, args.data.crop_size)
    audio_size = (args.data.audio_mels, int(args.data.audio_dur*64))
    encoder = VideoEarlyFusion(
        video_arch=args.model.video.backbone, video_pretrained=args.model.video.pretrained, video_size=video_size,
        audio_arch=args.model.audio.backbone, audio_pretrained=args.model.audio.pretrained, audio_size=audio_size,
        fusion_layers=args.model.fusion.layers,
        num_fusion_tkns=(args.model.fusion.num_fusion_tkns,
                         args.model.fusion.num_aggr_visual_tkns,
                         args.model.fusion.num_aggr_audio_tkns),
        drop_path=args.opt.drop_path,
        attn_drop=args.opt.attn_drop,
        drop=args.opt.proj_drop,
        fusion_mlp_ratio=args.model.fusion.mlp_ratio,
        fusion_attn_ratio=args.model.fusion.attn_ratio,
        fusion_num_heads=args.model.fusion.num_heads
    )
    model = AVClassifier(encoder=encoder, num_classes=1)
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
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        use_amp=args.opt.use_amp,
        accum_iter=args.opt.accum_iter,
        distributed=args.env.distributed
    )

    # Checkpointing
    ckpt_manager = misc_utils.CheckpointManager(
        modules=trainer.module_dict(),
        ckpt_dir=f"{job_dir}/checkpoints",
        epochs=args.opt.epochs,
        save_freq=args.log.save_freq)
    start_epoch = ckpt_manager.resume()[0] if args.opt.resume else 0
    wb_logger = misc_utils.WBLogger(
        f"{job_dir}/wandb", args.log.wandb_entity, args.log.wandb_project+'-avsync', args.job_name,
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
    header = f'[Train][Ep-{epoch}/{args.opt.epochs}]'
    metric_logger = meters.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', meters.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    trainer.zero_grad()
    for step, (video, audio, anno) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        sys.stdout.flush()
        global_step = (len(loader) // trainer.accum_iter) * epoch + step // trainer.accum_iter
        if step % args.opt.accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(trainer.optimizer, epoch + step / len(loader), args)
            metric_logger.update(lr=lr)

        # Prepare data
        video = video.to(device, non_blocking=True).float()
        audio = audio.to(device, non_blocking=True).float()
        targets = anno['sync'].to(device, non_blocking=True).float()

        # Forward pass
        with trainer.autocast(), trainer.autosync():
            preds_video, preds_audio, preds_fusion = trainer.model(video, audio)
            preds = (preds_video + preds_audio + preds_fusion).squeeze(1) / 3.
            loss = trainer.criterion(preds, targets)

        if not math.isfinite(loss.item()):
            raise f"Loss is {loss.item()}, stopping training"

        # Backward pass and model update
        grad_norm, amp_scale = trainer.step(loss)

        # Log
        if trainer.accums == 0:
            acc = ((preds > 0).float() == targets).float().mean() * 100.
            metric_logger.update(loss=loss.item(), train_sync_acc=acc.item(), grad_norm=grad_norm, amp_scale=amp_scale, n=video.shape[0])
            wb_logger.log(metric_logger.latest(), step=global_step)

        if args.debug and step == 8:
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

    for step, (video, audio, anno) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        # Prepare data
        video = video.to(device, non_blocking=True).float()
        audio = audio.to(device, non_blocking=True).float()
        targets = anno['sync'].to(device, non_blocking=True).float()

        # Compute sync predictions
        preds_video, preds_audio, preds_fusion = model(video, audio)
        preds = (preds_video + preds_audio + preds_fusion).squeeze(1) / 3.
        acc = ((preds>0).float() == targets).float().mean() * 100.
        metric_logger.update(test_sync_acc=acc, n=len(targets))

        if args.debug and step == 8:
            break

    metric_logger.synchronize_between_processes()
    return metric_logger.averages()
