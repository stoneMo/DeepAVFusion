import sys
import math
from typing import Iterable, Optional

import torch
from torch import nn

import datasets as myDBs
from torchvision import transforms as vT
from util import audio_transforms as aT
from timm.data.mixup import Mixup, one_hot

from models.deepavfusion import DeepAVFusion
from models.classifier import AVClassifier
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

from util import distributed as dist_utils
from util import misc as misc_utils
from util import data as data_utils
from util import meters, lr_sched


class AVMixup(Mixup):
    def _mix_batch(self, x):
        image, audio = x
        assert len(image) % 2 == 0, 'Batch size should be even when using this'
        assert self.cutmix_alpha == 0

        # lam, use_cutmix = self._params_per_batch()
        lam, use_cutmix = self._params_per_elem(image.shape[0])
        assert all([not u for u in use_cutmix])
        if all([l == 1. for l in lam]):
            return 1., (image, audio)
        lam = torch.from_numpy(lam).to(image.device)
        image_flipped = image.flip(0).mul_(1. - lam[:, None, None, None])
        image.mul_(lam[:, None, None, None]).add_(image_flipped)
        audio_flipped = audio.flip(0).mul_(1. - lam[:, None, None, None])
        audio.mul_(lam[:, None, None, None]).add_(audio_flipped)
        return lam, (image, audio)

    def mixup_target(self, target, lam=1.):
        off_value = self.label_smoothing / self.num_classes
        on_value = 1. - self.label_smoothing + off_value
        if target.ndim == 1:
            y1 = one_hot(target, self.num_classes, on_value=on_value, off_value=off_value)
            y2 = one_hot(target.flip(0), self.num_classes, on_value=on_value, off_value=off_value)
        else:
            y1 = (target == 1) * on_value + (target == 0) * off_value
            y2 = (target.flip(0) == 1) * on_value + (target.flip(0) == 0) * off_value
        return y1 * lam[:, None] + y2 * (1. - lam[:, None])

    def __call__(self, x, target):
        assert self.mode == 'batch'
        lam, x = self._mix_batch(x)
        target = self.mixup_target(target, lam)
        return x, target


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
        dataset_type='simple',
        visual_transform=vT.Compose([
            vT.RandomResizedCrop(args.data.image_size, scale=(args.data.crop_min, 1.)),
            vT.RandomHorizontalFlip(),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        audio_transform=aT.Compose([
            aT.Pad(rate=args.data.audio_rate, dur=args.data.audio_dur),
            aT.RandomVol(),
            aT.MelSpectrogram(sample_rate=args.data.audio_rate, n_fft=int(args.data.audio_rate * 0.05), hop_length=int(args.data.audio_rate / 64), n_mels=args.data.audio_mels),
            aT.Log()]),
        train=True,
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
        dataset_type='simple',
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
        audio_dur=args.data.audio_dur,
        audio_rate=args.data.audio_rate,
    )
    loader_val = data_utils.get_dataloader(
        dataset_val, args.env.distributed, args.opt.batch_size, args.env.workers, shuffle=False, drop_last=False)
    print(dataset_val)

    dataset_val_dense = myDBs.load_dataset(
        args.data.dataset,
        args.data.data_path,
        dataset_type='dense',
        visual_transform=vT.Compose([
            vT.Resize(int(args.data.image_size/0.875)),
            vT.CenterCrop(args.data.image_size),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        audio_transform=aT.Compose([
            aT.Pad(rate=args.data.audio_rate, dur=args.data.audio_dur),
            aT.RandomVol((-2, 2)),
            aT.MelSpectrogram(sample_rate=args.data.audio_rate, n_fft=int(args.data.audio_rate * 0.05), hop_length=int(args.data.audio_rate / 64), n_mels=args.data.audio_mels),
            aT.Log()]),
        train=False,
        audio_dur=args.data.audio_dur,
        audio_rate=args.data.audio_rate,
        dense_n=10,
        dense_span=10,
    )
    loader_val_dense = data_utils.get_dataloader(
        dataset_val_dense, args.env.distributed, max(args.opt.batch_size//4, 1), args.env.workers, shuffle=False, drop_last=False)
    print(dataset_val_dense)

    # MixUp
    mixup_fn = None
    mixup_active = args.data.mixup > 0 or args.data.cutmix > 0. or args.data.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = AVMixup(
            mixup_alpha=args.data.mixup, cutmix_alpha=args.data.cutmix, cutmix_minmax=args.data.cutmix_minmax,
            prob=args.data.mixup_prob, switch_prob=args.data.mixup_switch_prob, mode=args.data.mixup_mode,
            label_smoothing=args.opt.smoothing, num_classes=myDBs.NUM_CLASSES[args.data.dataset])

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
    model = AVClassifier(encoder, myDBs.NUM_CLASSES[args.data.dataset], freeze_encoder=False, input_norm=False)
    class_freq = dataset_train.class_dist
    bias_init = torch.log((class_freq + 1e-3)/(1 - class_freq + 1e-3))
    model.image_head.bias.data[:] = bias_init
    model.audio_head.bias.data[:] = bias_init
    model.fusion_head.bias.data[:] = bias_init
    model.to(device)
    print("Model = %s" % str(model))

    if args.checkpoint or args.pretrain_job_name:
        pretrain_ckpt = args.checkpoint or f"{args.output_dir}/checkpoints/checkpoint_{args.pretrain_resume_epoch}.pth"
        encoder.load_checkpoint(pretrain_ckpt, args.encoder_prefix)

    # criterion
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy() if not myDBs.MULTI_CLASS_DBS[args.data.dataset] else nn.BCEWithLogitsLoss(reduction='none')
    elif args.opt.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.opt.smoothing)
    else:
        criterion = nn.CrossEntropyLoss() if not myDBs.MULTI_CLASS_DBS[args.data.dataset] else nn.BCEWithLogitsLoss()
    print("criterion = %s" % str(criterion))

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lr_sched.param_groups_lrd(
        model, args.opt.weight_decay,
        no_weight_decay_list=[n for n, p in model.named_parameters() if 'bias' in n or 'norm' in n],
        layer_decay=args.opt.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.opt.lr)

    # Trainer
    trainer = misc_utils.Trainer(
        model,
        criterion=criterion,
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
        f"{job_dir}/wandb", args.log.wandb_entity, args.log.wandb_project+'-finetune', args.job_name,
        model, args)

    if args.eval:
        # Eval using 10 frames per sample
        evaluate(trainer.eval_model, loader_val_dense, start_epoch, device, args)
        exit(0)

    # =============================================================== #
    # Training loop
    print(f"Start training for {args.opt.epochs} epochs")
    for epoch in range(start_epoch, args.opt.epochs):
        if args.env.distributed:
            loader_train.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(trainer, loader_train, mixup_fn, epoch,
                        device=device, wb_logger=wb_logger, args=args)

        # evaluate
        if epoch % args.log.eval_freq == 0 or epoch == args.opt.epochs - 1 or epoch == start_epoch:
            global_step = (len(loader_train) // trainer.accum_iter) * (epoch + 1)
            test_stats = evaluate(trainer.eval_model, loader_val, epoch, device, args)
            wb_logger.log(test_stats, step=global_step, force=True)

        # save checkpoint
        ckpt_manager.checkpoint(epoch+1, {'epoch': epoch+1})

    # Eval using 10 frames per sample
    evaluate(trainer.eval_model, loader_val_dense, args.opt.epochs, device, args)


def train_one_epoch(trainer: misc_utils.Trainer,
                    loader: Iterable,
                    mixup_fn: Optional[AVMixup],
                    epoch: int = 0,
                    wb_logger: misc_utils.WBLogger = None,
                    device: torch.device = torch.device('cpu'),
                    args=None):
    trainer.model.train(True)
    metric_logger = meters.MetricLogger(delimiter="  ")
    header = f'[Train][Ep-{epoch}/{args.opt.epochs}]'

    trainer.zero_grad()
    class_freq = loader.dataset.class_dist.to(device)
    for step, (image, audio, anno) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        sys.stdout.flush()
        global_step = (len(loader) // trainer.accum_iter) * epoch + step // trainer.accum_iter
        if step % args.opt.accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(trainer.optimizer, epoch + step / len(loader), args)
            metric_logger.update(lr=lr)

        # Prepare data
        image = image.to(device, non_blocking=True).float()
        audio = audio.to(device, non_blocking=True).float()
        targets = anno['class'].to(device, non_blocking=True)
        targets = targets.float() if myDBs.MULTI_CLASS_DBS[args.data.dataset] else targets.long()
        if mixup_fn is not None:
            (image, audio), targets = mixup_fn((image, audio), targets)

        # Forward pass
        with trainer.autocast(), trainer.autosync():
            preds_image, preds_audio, preds_fusion = trainer.model(image, audio)
            if args.opt.joint_loss:
                preds = (preds_image + preds_audio + preds_fusion) / 3.
                loss = trainer.criterion(preds[:, class_freq>0], targets[:, class_freq>0])
            else:
                loss_image = trainer.criterion(preds_image[:, class_freq>0], targets[:, class_freq>0])
                loss_audio = trainer.criterion(preds_audio[:, class_freq>0], targets[:, class_freq>0])
                loss_fusion = trainer.criterion(preds_fusion[:, class_freq>0], targets[:, class_freq>0])
                loss = (loss_image + loss_audio + loss_fusion) / 3.

            loss = (loss / class_freq[None, class_freq>0]).mean()

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

    preds_all, preds_image_all, preds_audio_all, preds_fusion_all, labels_all = [], [], [], [], []
    for step, (image, audio, anno) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        image = image.to(device, non_blocking=True).float()
        audio = audio.to(device, non_blocking=True).float()
        label = anno['class'].to(device, non_blocking=True).long()

        if image.ndim == 5:
            preds_image, preds_audio, preds_fusion = model(image.flatten(0, 1), audio.flatten(0, 1))
            (B, N), C = image.shape[:2], preds_image.shape[-1]
            preds_image = preds_image.reshape(B, N, C).mean(1)
            preds_audio = preds_audio.reshape(B, N, C).mean(1)
            preds_fusion = preds_fusion.reshape(B, N, C).mean(1)
        else:
            preds_image, preds_audio, preds_fusion = model(image, audio)
        preds = (preds_image + preds_audio + preds_fusion) / 3.
        preds_image_all.append(preds_image), preds_audio_all.append(preds_audio), preds_fusion_all.append(preds_fusion)
        preds_all.append(preds), labels_all.append(label)
        metric_logger.update(
            recall_image=(preds_image[label==1] > 0).float().mean() * 100.,
            recall_audio=(preds_audio[label==1] > 0).float().mean() * 100.,
            recall_fusion=(preds_fusion[label==1] > 0).float().mean() * 100.,
            recall_all=(preds[label==1] > 0).float().mean() * 100.,
            n=label.shape[0])

        if args.debug and step == 8:
            break

    # Synchronize across gpus
    preds_image_all = dist_utils.concat_all_gather(torch.cat(preds_image_all))
    preds_audio_all = dist_utils.concat_all_gather(torch.cat(preds_audio_all))
    preds_fusion_all = dist_utils.concat_all_gather(torch.cat(preds_fusion_all))
    preds_all = dist_utils.concat_all_gather(torch.cat(preds_all))
    labels_all = dist_utils.concat_all_gather(torch.cat(labels_all))

    # measure performance
    stats = dict()
    if myDBs.MULTI_CLASS_DBS[args.data.dataset]:
        labels_all = labels_all.cpu().numpy()

        for mod, preds in [('image', preds_image_all), ('audio', preds_audio_all), ('fusion', preds_fusion_all), ('all', preds_all)]:
            preds = preds.cpu().numpy()
            stats_mod = misc_utils.calc_multi_class_stats(labels_all, preds)
            stats.update({f'{k}_{mod}': v for k, v in stats_mod.items()})
    else:
        stats.update(
            acc1_image=accuracy(preds_image_all, labels_all)[0].item(),
            acc1_audio=accuracy(preds_audio_all, labels_all)[0].item(),
            acc1_fusion=accuracy(preds_fusion_all, labels_all)[0].item(),
            acc1_all=accuracy(preds_all, labels_all)[0].item(),
        )
    prefix = 'val_' if image.ndim == 4 else 'val_dense_'
    stats = {f"{prefix}{k}": v for k, v in stats.items()}

    msg = ' | '.join([f'{k}={v:.2f}' for k, v in stats.items()])
    print(f"{header} {msg}")
    return stats
