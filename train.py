import sys
import math
from typing import Iterable
import torch

import datasets as myDBs
from torchvision import transforms as vT
from util import audio_transforms as aT

from models.deepavfusion import DeepAVFusion
from models.avmae import AVMAE

from util import distributed as dist_utils
from util import misc as misc_utils
from util import data as data_utils
from util import meters, lr_sched
from util.knn_probe import EvalAVNNProbe


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
    dataset = myDBs.load_dataset(
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
        temporal_jitter=True,
    )
    loader = data_utils.get_dataloader(
        dataset, args.env.distributed, args.opt.batch_size, args.env.workers, shuffle=True, drop_last=True)
    print(dataset)

    # Create model
    image_size, audio_size = (args.data.image_size, args.data.image_size), (args.data.audio_mels, int(args.data.audio_dur * 64))
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
    model = AVMAE(
        encoder, encoder.embed_dim,
        image_decoder_arch=args.model.image.decoder_arch, image_decoder_depth=args.model.image.decoder_depth,
        image_mask_ratio=args.model.image.mask_ratio, image_norm_loss=args.model.image.norm_loss,
        audio_decoder_arch=args.model.audio.decoder_arch, audio_decoder_depth=args.model.audio.decoder_depth,
        audio_mask_ratio=args.model.audio.mask_ratio, audio_norm_loss=args.model.audio.norm_loss
    )
    model.to(device)
    print("Model = %s" % str(model))

    # Optimizer
    no_weight_decay_list = [n for n, p in model.named_parameters() if 'bias' in n or 'norm' in n]
    param_groups = lr_sched.param_groups_pretrained(
        model, args.opt.weight_decay, no_weight_decay_list=no_weight_decay_list,
        image_pt=args.model.image.pretrained, audio_pt=args.model.audio.pretrained)
    optimizer = torch.optim.AdamW(param_groups, lr=args.opt.lr, betas=(0.9, 0.95))
    print(optimizer)

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
        f"{job_dir}/wandb", args.log.wandb_entity, args.log.wandb_project, args.job_name,
        model, args)

    # Set up probes
    knn_probe = EvalAVNNProbe(args.nn_probe, args.log, args.env)

    # =============================================================== #
    # Training loop
    print(f"Start training for {args.opt.epochs} epochs")
    for epoch in range(start_epoch, args.opt.epochs):
        if args.env.distributed:
            loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(loader, trainer, epoch,
                        device=device, wb_logger=wb_logger, args=args)

        # evaluate
        if epoch % args.log.eval_freq == 0 or epoch == args.opt.epochs - 1 or epoch == start_epoch:
            global_step = (len(loader) // trainer.accum_iter) * (epoch + 1)
            knn_stats = knn_probe.evaluate(trainer.eval_model, epoch=epoch)
            wb_logger.log(knn_stats, step=global_step, force=True)

        # save checkpoint
        ckpt_manager.checkpoint(epoch+1, {'epoch': epoch+1})


def train_one_epoch(loader: Iterable,
                    trainer: misc_utils.Trainer,
                    epoch: int = 0,
                    wb_logger: misc_utils.WBLogger = None,
                    device: torch.device = torch.device('cpu'),
                    args=None):
    trainer.model.train(True)
    metric_logger = meters.MetricLogger(delimiter="  ")
    header = f'[Train][Ep-{epoch}/{args.opt.epochs}]'

    trainer.zero_grad()
    for step, (image, audio, _) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        sys.stdout.flush()
        global_step = (len(loader) // trainer.accum_iter) * epoch + step // trainer.accum_iter
        if step % args.opt.accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(trainer.optimizer, epoch + step / len(loader), args)
            metric_logger.update(lr=lr)

        # Prepare data
        image = image.to(device, non_blocking=True).float()
        audio = audio.to(device, non_blocking=True).float()

        # Forward pass
        with trainer.autocast(), trainer.autosync():
            loss_image, loss_audio = trainer.model(image, audio)[:2]
            loss = loss_image + loss_audio
        if not math.isfinite(loss.item()):
            raise f"Loss is {loss.item()}, stopping training"

        # Backward pass and model update
        grad_norm, amp_scale = trainer.step(loss)

        # Log
        if trainer.accums == 0:
            metric_logger.update(
                loss=loss.item(), loss_image=loss_image.item(), loss_audio=loss_audio.item(),
                grad_norm=grad_norm, amp_scale=amp_scale, n=image.shape[0])
            wb_logger.log(metric_logger.latest(), step=global_step)

        if args.debug and step == 100:
            break

    # gather the stats from all processes
    print("Syncing meters...")
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    trainer.zero_grad()
    return metric_logger.averages()
