import sys
import math
from typing import Iterable

import torch
import torch.nn as nn

import datasets as myDBs
from util import image_labels_transforms as vT
from util import audio_transforms as aT

from models.deepavfusion import DeepAVFusion
from models.avsegm import AVSegmSimple

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
        visual_transform=vT.Compose([
            vT.RandomResizedCrop(args.data.image_size, scale=(args.data.crop_min, 1.)),
            vT.RandomHorizontalFlip(),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        audio_transform=aT.Compose([
            aT.Pad(rate=args.data.audio_rate, dur=args.data.audio_dur),
            aT.RandomVol(),
            aT.MelSpectrogram(sample_rate=args.data.audio_rate, n_fft=int(args.data.audio_rate * 0.05), hop_length=int(args.data.audio_rate / 64), n_mels=args.data.audio_mels),
            aT.Log()
        ]),
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
        visual_transform=vT.Compose([
            vT.Resize(args.data.image_size),
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
        temporal_jitter=False,
        dense=True,
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
    num_classes = myDBs.NUM_CLASSES[args.data.dataset]
    model = AVSegmSimple(encoder=encoder, num_classes=num_classes if num_classes > 2 else 1)
    model.to(device)
    print("Model = %s" % str(model))

    if args.checkpoint or args.pretrain_job_name:
        pretrain_ckpt = args.checkpoint or f"{args.output_dir}/checkpoints/checkpoint_{args.pretrain_resume_epoch}.pth"
        encoder.load_checkpoint(pretrain_ckpt, args.encoder_prefix)

    # build optimizer with layer-wise lr decay (lrd)
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
        f"{job_dir}/wandb", args.log.wandb_entity, args.log.wandb_project+'-avsegm', args.job_name,
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
    for step, (image, audio, anno, _) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        sys.stdout.flush()
        global_step = (len(loader) // trainer.accum_iter) * epoch + step // trainer.accum_iter
        if step % args.opt.accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(trainer.optimizer, epoch + step / len(loader), args)
            metric_logger.update(lr=lr)

        # Prepare data
        image = image.to(device, non_blocking=True).float()
        audio = audio.to(device, non_blocking=True).float()
        gt_mask = anno['gt_map'].cuda(device, non_blocking=True)

        # Forward pass
        with trainer.autocast(), trainer.autosync():
            loss, _ = trainer.model(image, audio, gt_mask)

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

    evaluator = AVSegmEvaluator(device=device)
    for step, (image_b, audio_b, anno_b, _) in enumerate(metric_logger.log_every(loader, args.log.print_freq, header)):
        for idx in range(image_b.shape[0]):
            # prepare data
            image = image_b[idx].to(device, non_blocking=True).float()
            audio = audio_b[idx].to(device, non_blocking=True).float()
            gt_mask = anno_b['gt_map'][idx].cuda(device, non_blocking=True)

            # forward
            _, pred = model(image, audio, gt_mask)

            # measure iou and fscore
            evaluator.update(pred, gt_mask)

        if args.debug and step == 8:
            break

    metrics = evaluator.average_metrics()
    print(f"Averaged stats: miou={metrics['miou']:.3f} miou_noBg={metrics['miou_noBg']:.3f} fscore={metrics['fscore']:.3f} fscore_noBg={metrics['fscore_noBg']:.3f}")
    return metrics


class AVSegmEvaluator:
    def __init__(self, beta2=0.3, device=torch.device('cpu')):
        self.beta2 = beta2
        self.device = device
        self.clear()

    def clear(self):
        self.iou_list = []
        self.cls_list = []
        self.precision_list = []
        self.recall_list = []
        self.fscore_list = []

    def update(self, pred, target):
        nclass = pred.shape[1]
        target = target.cpu().int() + 1
        if pred.shape[1] == 1:
            prob = torch.sigmoid(pred).cpu()
            pred = (prob.squeeze(1) > 0.5).float() + 1
        else:
            prob = torch.softmax(pred.cpu(), dim=1)
            pred = torch.argmax(prob, 1).float() + 1
        prob *= (target.unsqueeze(1) > 0).float()
        pred *= (target > 0).float()

        for y, p, t in zip(pred, prob, target):
            if (t > 1).sum() == 0.0:    # Ignore samples with only background
                continue
            if nclass == 1:
                iou, precision, recall, _, cls = self._batch_miou_fscore(y, t, 2)
                fscore = self._batch_fscore_bin(p.squeeze(0), t)
            else:
                iou, precision, recall, fscore, cls = self._batch_miou_fscore(y, t, nclass)
            self.iou_list.append(iou)
            self.cls_list.append(cls)
            self.precision_list.append(precision)
            self.recall_list.append(recall)
            self.fscore_list.append(fscore)

    def _batch_fscore_bin(self, prob, target, eps=1e-10, nbins=256):
        prec, recall = torch.zeros(nbins), torch.zeros(nbins)
        for i, thr in enumerate(torch.linspace(0, 1 - eps, nbins)):
            ypred = (prob >= thr).int() + 1
            tp = ((ypred == 2).int() * (target == 2).int()).sum()
            prec[i] = tp / ((ypred == 2).int().sum() + eps)
            recall[i] = tp / ((target == 2).int().sum() + eps)
        f_score = (1 + self.beta2) * prec * recall / (self.beta2 * prec + recall + eps)
        return f_score.max()

    def _batch_miou_fscore(self, pred, target, nclass, eps=1e-10):
        intersection = pred * (pred == target) # [BF, H, W]
        area_inter = torch.histc(intersection.float(), bins=nclass, min=1, max=nclass)  # TP
        area_pred = torch.histc(pred.float(), bins=nclass, min=1, max=nclass) # TP + FP
        area_lab = torch.histc(target.float(), bins=nclass, min=1, max=nclass) # TP + FN
        area_union = area_pred + area_lab - area_inter

        iou = area_inter / (eps + area_union)
        cls_count = torch.eye(nclass)[torch.nonzero(area_union).squeeze(-1)].sum(0)

        precision = area_inter / (eps + area_pred)
        recall = area_inter / (eps + area_lab)
        fscore = (1 + self.beta2) * precision * recall / (self.beta2 * precision + recall + eps)

        return iou, precision, recall, fscore, cls_count

    def _aggregate_metric(self, x_list):
        x_sum = torch.stack(x_list).sum(0)
        if dist_utils.is_dist_avail_and_initialized() and self.device != 'cpu':
            x_sum = dist_utils.concat_all_gather(x_sum[None].to(self.device)).sum(0)
        return x_sum

    def average_metrics(self):
        cls_sum = self._aggregate_metric(self.cls_list)
        iou_sum = self._aggregate_metric(self.iou_list)
        precision_sum = self._aggregate_metric(self.precision_list)
        recall_sum = self._aggregate_metric(self.recall_list)
        fscore_sum = self._aggregate_metric(self.fscore_list)

        seen_classes = torch.nonzero(cls_sum).squeeze(1)
        miou_per_cls = iou_sum[seen_classes] / cls_sum[seen_classes]
        precision_per_cls = precision_sum[seen_classes] / cls_sum[seen_classes]
        recall_per_cls = recall_sum[seen_classes] / cls_sum[seen_classes]
        if fscore_sum.numel() == 1: # FScore for Binary Segmentation
            num_images = len(self.fscore_list) * dist_utils.get_world_size()
            fscore_per_cls = torch.stack([fscore_sum, fscore_sum]) / num_images
        else:
            fscore_per_cls = fscore_sum[seen_classes] / cls_sum[seen_classes]

        return {
            'miou': torch.mean(miou_per_cls).item(),
            'miou_noBg': torch.mean(miou_per_cls[1:]).item(),
            'precision': torch.mean(precision_per_cls).item(),
            'precision_noBg': torch.mean(precision_per_cls[1:]).item(),
            'recall': torch.mean(recall_per_cls).item(),
            'recall_noBg': torch.mean(recall_per_cls[1:]).item(),
            'fscore': torch.mean(fscore_per_cls).item(),
            'fscore_noBg': torch.mean(fscore_per_cls[1:]).item(),
        }
