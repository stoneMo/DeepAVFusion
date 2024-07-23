import os
import sys
import copy

import numpy as np
from collections import defaultdict

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from datasets import get_vggsound, get_audioset
import torch.utils.data
from sklearn import metrics


from util import distributed as dist_utils
from util import meters
from torch.nn import functional as F
from torchvision import transforms as vT
from util import audio_transforms as aT


class EvalAVNNProbe:
    def __init__(self, probe_args, log_args, env_args):
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.distributed = env_args.distributed
        self.eval_freq = log_args.eval_freq
        self.print_freq = log_args.print_freq
        self.dataset = probe_args.dataset

        image_transform = vT.Compose([
            vT.Resize(int(probe_args.image_size/0.875)),
            vT.CenterCrop(probe_args.image_size),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        audio_transform = vT.Compose([
            aT.Pad(rate=probe_args.audio_rate, dur=probe_args.audio_dur),
            aT.MelSpectrogram(sample_rate=probe_args.audio_rate, n_fft=int(probe_args.audio_rate * 0.05), hop_length=int(probe_args.audio_rate / 64), n_mels=probe_args.audio_mels),
            aT.Log(),
        ])

        if self.dataset == 'vggsound':
            self.db = get_vggsound(
                probe_args.data_path,
                partition='test',
                audio_dur=probe_args.audio_dur,
                audio_rate=probe_args.audio_rate,
                visual_transform=image_transform,
                audio_transform=audio_transform
            )
            self.multi_label = False
        elif self.dataset == 'audioset':
            self.db = get_audioset(
                probe_args.data_path,
                partition='eval',
                audio_dur=probe_args.audio_dur,
                audio_rate=probe_args.audio_rate,
                visual_transform=image_transform,
                audio_transform=audio_transform
            )
            self.multi_label = True
        else:
            raise NotImplementedError

        if self.distributed:
            num_tasks = dist_utils.get_world_size()
            global_rank = dist_utils.get_rank()
            self.sampler = torch.utils.data.DistributedSampler(
                self.db, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            self.sampler = torch.utils.data.RandomSampler(self.db)
        self.loader = torch.utils.data.DataLoader(
            self.db,
            sampler=self.sampler,
            batch_size=max(probe_args.batch_size//4, 1),
            num_workers=max(env_args.workers, 1),
            pin_memory=False,
            drop_last=True,
        )

    @torch.no_grad()
    def evaluate(self, model, epoch=0):
        # model = copy.deepcopy(model)
        model.train(False)
        if self.distributed:
            self.sampler.set_epoch(0)

        # Extract features
        a_feats, v_feats, mm_feats, labels = [], [], [], []
        metric_logger = meters.MetricLogger(delimiter="  ")
        for image, spec, anno in metric_logger.log_every(self.loader, self.print_freq, 'Extract features'):
            # Prepare data
            spec = spec.to(self.device, non_blocking=True).float()
            image = image.to(self.device, non_blocking=True).float()
            lbl = anno['class'].to(self.device, non_blocking=True).long()

            # Extract features
            x_v, x_a, x_mm = model.forward_encoder(image, spec)[:3]

            # Collect features and labels
            v_feats.append(x_v.mean(dim=1))
            a_feats.append(x_a.mean(dim=1))
            mm_feats.append(x_mm.mean(dim=1))
            labels.append(lbl)

        # Synchronize across gpus
        a_feats = dist_utils.concat_all_gather(F.normalize(torch.cat(a_feats), p=2, dim=1))
        v_feats = dist_utils.concat_all_gather(F.normalize(torch.cat(v_feats), p=2, dim=1))
        mm_feats = dist_utils.concat_all_gather(F.normalize(torch.cat(mm_feats), p=2, dim=1))
        labels = dist_utils.concat_all_gather(torch.cat(labels))
        n_data = labels.shape[0]

        # kNN Evaluation
        metric_logger = meters.MetricLogger(delimiter="  ")
        preds = defaultdict(list)
        for i in metric_logger.log_every(range(0, n_data, 128), 250):
            qa_feats = a_feats[i:i+128]
            qv_feats = v_feats[i:i+128]
            qmm_feats = mm_feats[i:i+128]

            # Find nearest neighbor and aggregate predictions
            scores_a = torch.einsum('qd,nd->qn', qa_feats, a_feats)
            scores_v = torch.einsum('qd,nd->qn', qv_feats, v_feats)
            scores_mm = torch.einsum('qd,nd->qn', qmm_feats, mm_feats)

            for mod, scores in [('audio', scores_a), ('image', scores_v), ('fusion', scores_mm), ('all', scores_v+scores_a+scores_mm)]:
                scores, nn_idx = torch.topk(scores, k=2, dim=1, sorted=True)
                preds[mod].append((
                    labels[nn_idx[:, 1]],
                    scores[:, 1]
                ))

        # Compute accuracies/auc
        metrics_dict = {}
        labels = labels.cpu().numpy()
        if self.multi_label:
            seen_classes = labels.sum(0) > 0
            for mod in preds:
                scores = torch.cat([(ypred * yscore[:, None]) for ypred, yscore in preds[mod]]).cpu().numpy()
                ap = metrics.average_precision_score(labels[:, seen_classes], scores[:, seen_classes], average=None)
                auc = metrics.roc_auc_score(labels[:, seen_classes], scores[:, seen_classes], average=None)
                metrics_dict.update({f'{mod}_nn_ap': ap.mean(), f'{mod}_nn_auc': auc.mean()})

        else:
            for mod in preds:
                ypred = torch.cat([ypred  for ypred, _ in preds[mod]]).cpu().numpy()
                acc = np.mean(ypred == labels)*100
                metrics_dict.update({f'{mod}_nn_acc': acc})

        print(metrics_dict)
        return metrics_dict

