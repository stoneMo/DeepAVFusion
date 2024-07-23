import builtins
import datetime
import sys
import random
import numpy as np
import warnings

import torch
import torch.distributed as dist
from torch.backends import cudnn


def setup_for_distributed(is_master, log_fn=None):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_rank() % 8 == 0)
        if is_master or force:
            # print with time stamp
            now = datetime.datetime.now().time()
            msg = f'[{now}] ' + ' '.join([str(ct) for ct in args])

            # Print to terminal
            builtin_print(msg, **kwargs)
            sys.stdout.flush()

            # Log to file
            if log_fn is not None:
                open(log_fn, 'a').write(msg + '\n')

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(local_rank, args, log_fn):
    ngpus_per_node = torch.cuda.device_count()
    args.env.distributed = ngpus_per_node > 0
    if args.env.distributed:
        args.env.world_size = ngpus_per_node * args.env.world_size
        args.env.rank = args.env.rank * ngpus_per_node + local_rank
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True, log_fn=log_fn)  # hack
        args.env.world_size = 1
        args.env.rank = 0
        return

    dist.init_process_group(backend='nccl', init_method=args.env.dist_url,
                            world_size=args.env.world_size, rank=args.env.rank)

    torch.cuda.set_device(local_rank)
    print('Distributed init (rank {}): {}, gpu {}'.format(
        args.env.rank, args.env.dist_url, local_rank), flush=True)
    print('before barrier')
    torch.distributed.barrier()
    print('after barrier')
    setup_for_distributed(args.env.rank == 0, log_fn=log_fn)

    if args.env.seed is not None:
        seed = args.env.seed + get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def all_gather(obj):
    gather_list = [None] * get_world_size()
    dist.all_gather_object(gather_list, obj)
    return gather_list