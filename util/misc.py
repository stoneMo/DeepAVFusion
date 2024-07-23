# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import os
import math

import torch
import torch.nn as nn
import torch.utils.data

from util import distributed as dist_utils
import numpy as np
from sklearn import metrics
from collections import deque
import copy

import wandb
import contextlib


def print_args(args, prefix=''):
    from omegaconf.dictconfig import DictConfig
    for k in args:
        if isinstance(args[k], DictConfig):
            print_args(args[k], prefix=prefix + f'{k}.')
        else:
            print(f"{prefix}{k}: {str(args[k])}")


class Trainer:
    def __init__(self, model, criterion=None, optimizer=None, accum_iter=1, use_amp=True, distributed=False):
        self.distributed = distributed
        self.model_without_ddp = model
        self.n_steps = torch.tensor([0])
        if self.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.accum_iter = accum_iter
        self.accums = 0

        self.eval_model = self.model_without_ddp
        self.zero_grad()

    def module_dict(self):
        d = {'state_dict': self.model_without_ddp, 'n_steps': self.n_steps}
        if self.criterion is not None:
            d['criterion'] = self.criterion
        if self.optimizer is not None:
            d['optimizer'] = self.optimizer
        if self.scaler is not None:
            d['scaler'] = self.scaler
        return d

    def zero_grad(self):
        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.grad_bkp = None
        self.accums = 0

    def get_scale(self):
        if self.scaler is not None:
            return self.scaler.get_scale()
        else:
            return 1.

    def backward(self, loss, create_graph=False):
        # Backward pass
        if self.scaler is not None:
            loss = self.scaler.scale(loss)

        loss.backward(create_graph=create_graph)
        self.accums += 1    # Track number of gradients accumulated

        # Track gradient norm (adjust by loss scale and # accumulated grads)
        norm = get_grad_norm_(self.model.parameters()) / (self.accums * self.get_scale())
        return norm.item(), self.get_scale()

    def backup_grads(self):
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.grad_bkp = param.grad
                param.grad = None

    def restore_grads(self):
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    param.grad = param.grad_bkp
                else:
                    param.grad += param.grad_bkp
                param.grad_bkp = None

    def step(self, loss, create_graph=False, clip_grad=None, skip_grad=None):
        # Backup grads in case norm too large
        if skip_grad is not None:
            self.backup_grads()
            norm, scale = self.backward(loss, create_graph=create_graph)
            if norm > skip_grad:
                self.optimizer.zero_grad()
                self.accums -= 1
            self.restore_grads()
        else:
            norm, scale = self.backward(loss, create_graph=create_graph)

        # Time to update?
        if self.accums == self.accum_iter:
            # Unscale gradients in-place
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Adjust for grad accumulation
            if self.accum_iter > 1:
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad /= self.accum_iter

            # Clip gradients in-place
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

            # Update parameters
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Reset grads
            self.zero_grad()
            self.n_steps += 1

        return norm, scale

    def autocast(self):
        if self.scaler is not None:
            return torch.cuda.amp.autocast()
        else:
            return contextlib.nullcontext()

    def autosync(self):
        if self.distributed and self.accums < self.accum_iter - 1:
            return self.model.no_sync()
        else:
            return contextlib.nullcontext()


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class WBLogger:
    def __init__(self, wandb_dir, entity, project, job_name, model, args):
        self.mute = not args.log.use_wandb or not dist_utils.is_main_process()
        if self.mute:
            return
        self.wandb_dir = wandb_dir
        self.entity = entity
        self.project = project
        self.job_name = job_name

        self.log_counter = 0   # Counter for spreading out logs
        self.log_freq = args.log.print_freq
        self.watch_freq = args.log.wandb_watch_freq

        self.init_wandb(args, model)

    def init_wandb(self, args, model):
        if self.mute:
            return

        os.makedirs(self.wandb_dir, exist_ok=True)
        runid = None
        if os.path.exists(f"{self.wandb_dir}/runid.txt"):
            runid = open(f"{self.wandb_dir}/runid.txt").read()
        wandb.init(entity=self.entity, project=self.project, name=self.job_name,
                   dir=self.wandb_dir, resume="allow", id=runid)
        open(f"{self.wandb_dir}/runid.txt", 'w').write(wandb.run.id)

        # Push config
        def flatten_args(args):
            from omegaconf import DictConfig
            args_flat = {}
            for k in dict(args).keys():
                if isinstance(args[k], DictConfig):
                    args_flat.update({f"{k}.{k2}": v for k2, v in flatten_args(args[k]).items()})
                else:
                    args_flat[k] = args[k]
            return args_flat
        wandb.config.update({k: v for k, v in flatten_args(args).items() if k not in wandb.config})

        # Log model stats

        if self.watch_freq > 0:
            wandb.watch(model, log="all", log_freq=self.watch_freq)

    def log(self, metrics, step=None, force=False):
        if self.mute:
            return
        if force or step is None:
            wandb.log(metrics, step=step)
        else:
            self.log_counter += 1
            if self.log_counter % self.log_freq == 0:
                wandb.log(metrics, step=step)


class CheckpointManager:
    def __init__(self,
                 modules,
                 ckpt_dir,
                 epochs,
                 save_freq=None):
        self.modules = modules
        self.ckpt_dir = ckpt_dir
        self.epochs = epochs
        self.save_freq = save_freq

        self.world_size = dist_utils.get_world_size()
        self.rank = dist_utils.get_rank()

        if self.rank == 0:
            os.makedirs(os.path.join(self.ckpt_dir), exist_ok=True)

    def map_location(self, state, device):
        if isinstance(state, dict):
            return {k: self.map_location(state[k], device) for k in state}
        elif isinstance(state, list):
            return [self.map_location(st, device) for st in state]
        elif isinstance(state, tuple):
            return tuple(self.map_location(st, device) for st in state)
        elif isinstance(state, torch.Tensor):
            return state.to(device)
        else:
            return state

    def create_state_dict(self, save_dict):
        state = {}
        for k in self.modules:
            if self.modules[k] is None:
                state[k] = None
            elif isinstance(self.modules[k], torch.Tensor):
                state[k] = copy.deepcopy(self.modules[k]).cpu()
            else:
                module_state = copy.deepcopy(self.modules[k].state_dict())
                state[k] = self.map_location(module_state, 'cpu')

        if save_dict is not None:
            state.update(save_dict)
        return state

    def load_state_dict(self, checkpoint):
        for k in self.modules:
            self.modules[k].load_state_dict(checkpoint[k])
        metrics = {k: checkpoint[k] for k in checkpoint if k not in self.modules}
        return metrics

    def resume(self):
        ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_latest.pth')
        print(f"Loading {ckpt_fname}")
        start_epoch, metrics = 0, {}
        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')

            # Load state dict
            for k in self.modules:
                if self.modules[k] is None:
                    continue
                elif isinstance(self.modules[k], torch.Tensor):
                    self.modules[k].data[:] = checkpoint[k].data
                else:
                    self.modules[k].load_state_dict(checkpoint[k])
            start_epoch = checkpoint['epoch']
            metrics = {k: checkpoint[k] for k in checkpoint if k not in set(self.modules.keys()) and k != 'epoch'}
            print(f"=> loaded checkpoint '{ckpt_fname}' (epoch {checkpoint['epoch']})")

        return start_epoch, metrics

    def checkpoint(self, epoch, save_dict=None, is_best=False):
        if self.rank != 0:
            return
        state = self.create_state_dict(save_dict)
        ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_latest.pth')
        torch.save(state, ckpt_fname)
        print(f"=> saved checkpoint '{ckpt_fname}' (epoch {epoch})")

        if is_best:
            best_fname = os.path.join(self.ckpt_dir, f'checkpoint_best.pth')
            torch.save(state, best_fname)
            print(f"=> saved best checkpoint '{best_fname}' (epoch {epoch})")

        if self.save_freq is not None and ((epoch % self.save_freq == 0) or epoch == self.epochs):
            ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save(state, ckpt_fname)
            print(f"=> saved checkpoint '{ckpt_fname}' (epoch {epoch})")


def calc_multi_class_stats(labels, preds):
    assert labels.shape[0] == preds.shape[0]

    seen_classes = labels.sum(0) > 0
    labels, preds = labels[:, seen_classes], preds[:, seen_classes]
    num_classes = seen_classes.sum()

    ap = np.array([metrics.average_precision_score(labels[:, cls], preds[:, cls], average=None)
                   for cls in range(num_classes)])
    auc = np.array([metrics.roc_auc_score(labels[:, cls], preds[:, cls], average=None)
                    for cls in range(num_classes)])

    return dict(
        ap=ap.mean()*100.,
        auc=auc.mean()*100.,
    )