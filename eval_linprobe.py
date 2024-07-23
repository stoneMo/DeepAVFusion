import sys
import math
from typing import Iterable

import torch
import torch.nn as nn

import datasets as myDBs
from torchvision import transforms as vT
from util import audio_transforms as aT

from models.deepavfusion import DeepAVFusion
from models.classifier import AVClassifier

from util import distributed as dist_utils
from util import misc as misc_utils
from util import data as data_utils
from util import meters, lars, lr_sched
from timm.utils import accuracy


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
        temporal_jitter=False
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
        fusion_mlp_ratio=args.model.fusion.mlp_ratio,
        fusion_attn_ratio=args.model.fusion.attn_ratio,
        fusion_num_heads=args.model.fusion.num_heads
    )
    model = AVClassifier(encoder, myDBs.NUM_CLASSES[args.data.dataset], freeze_encoder=True, input_norm=True)
    model.to(device)
    print("Model = %s" % str(model))

    if args.checkpoint or args.pretrain_job_name:
        pretrain_ckpt = args.checkpoint or f"{args.output_dir}/checkpoints/checkpoint_{args.pretrain_resume_epoch}.pth"
        encoder.load_checkpoint(pretrain_ckpt, args.encoder_prefix)

    # criterion
    criterion = nn.BCEWithLogitsLoss() if myDBs.MULTI_CLASS_DBS[args.data.dataset] else nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    # Optimizer
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 6
    optimizer = lars.LARS(parameters, lr=args.opt.lr, weight_decay=args.opt.weight_decay)

    # Trainer
    trainer = misc_utils.Trainer(
        model,
        criterion=criterion,
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
        f"{job_dir}/wandb", args.log.wandb_entity, args.log.wandb_project+'-linprobe', args.job_name,
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

        # Forward pass
        with trainer.autocast(), trainer.autosync():
            preds_image, preds_audio, preds_fusion = trainer.model(image, audio)
        preds = (preds_image + preds_audio + preds_fusion) / 3.
        loss = trainer.criterion(preds, targets)
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

        preds_image, preds_audio, preds_fusion = model(image, audio)
        preds = (preds_image + preds_audio + preds_fusion) / 3.
        preds_image_all.append(preds_image), preds_audio_all.append(preds_audio), preds_fusion_all.append(preds_fusion)
        preds_all.append(preds), labels_all.append(label)

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
            val_acc1_image=accuracy(preds_image_all, labels_all)[0].item(),
            val_acc1_audio=accuracy(preds_audio_all, labels_all)[0].item(),
            val_acc1_fusion=accuracy(preds_fusion_all, labels_all)[0].item(),
            val_acc1_all=accuracy(preds_all, labels_all)[0].item(),
        )
    prefix = 'val_'
    stats = {f"{prefix}{k}": v for k, v in stats.items()}

    msg = ' | '.join([f'{k}={v:.2f}' for k, v in stats.items()])
    print(f"{header} {msg}")
    return stats
