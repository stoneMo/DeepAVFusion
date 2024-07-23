#!/usr/bin/env python
import os
import copy
import time
import warnings
import logging
from pathlib import Path

import hydra
import hydra.utils as hydra_utils
import submitit

import torch
import torch.multiprocessing as mp
import importlib
import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

log = logging.getLogger(__name__)


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
    log.info('PYTHONPATH: {}'.format(os.environ["PYTHONPATH"]))


class Worker:
    def __call__(self, args):
        mp.set_start_method('spawn')
        main_function = getattr(importlib.import_module(args.worker), 'main_worker')

        np.set_printoptions(precision=3)
        socket_name = os.popen("ip r | grep default | awk '{print $5}'").read().strip('\n')
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name

        # Use random port to avoid collision between parallel jobs
        # if args.env.world_size == 1:
        #     args.env.port = np.random.randint(40000, 50000)
        #
        # if args.env.slurm:
        #     job_env = submitit.JobEnvironment()
        #     args.env.rank = job_env.global_rank
        #     args.env.dist_url = f'tcp://{job_env.hostnames[0]}:{args.env.port}'
        # else:
        #     args.env.rank = 0
        #     args.env.dist_url = f'tcp://localhost:{args.env.port}'
        # print('Using url {}'.format(args.env.dist_url))

        if args.env.slurm:
            job_env = submitit.JobEnvironment()
            args.env.rank = job_env.global_rank

        if args.env.ngpu > 1:
            os.environ['NCCL_P2P_DISABLE'] = '1'
            os.environ['NCCL_P2P_LEVEL'] = 'LOC'
            os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET6'
            args.env.dist_url = f"file://{args.output_dir}/{args.job_name}/.dist"
            os.makedirs(os.path.dirname(args.env.dist_url[7:]), exist_ok=True)
            time.sleep(2)

        if args.env.gpu is not None:
            warnings.warn(
                'You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

        # Run code
        ngpus_per_node = torch.cuda.device_count()
        args.env.distributed = args.env.world_size > 1 or (args.env.distributed and ngpus_per_node > 1)
        if args.env.distributed:
            mp.spawn(main_function, nprocs=ngpus_per_node, args=(args,))
        else:
            main_function(0, args)

    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(Worker(), *args, **kwargs)  # submits to requeuing


def my_jobs():
    return os.popen('squeue -o %j -u $USER').read().split("\n")


@hydra.main(config_path='configs/', config_name='efav', version_base='1.1')
def main(args):
    update_pythonpath_relative_hydra()
    args.output_dir = hydra_utils.to_absolute_path(args.output_dir)
    args.job_name = str(args.job_name)  # Resolve job name, before config vars change
    if 'pretrain_job_name' in args:     # If eval, fix paths
        args.output_dir = f"{args.output_dir}/{args.pretrain_job_name}"
    os.makedirs(f"{args.output_dir}/{args.job_name}", exist_ok=True)

    # defaults
    if args.env.workers is None:
        args.env.workers = 15 * args.env.ngpu
    if args.env.mem_gb is None:
        args.env.mem_gb = 60 * args.env.ngpu

    # If job is running, ignore
    if args.env.slurm:
        slurm_job_name = f"{args.job_name}-{args.pretrain_job_name}" if 'pretrain_job_name' in args and args.pretrain_job_name else args.job_name
        if slurm_job_name in my_jobs():
            print(f'Skipping {args.job_name} because already in queue')
            return

        # Submit jobs
        executor = submitit.AutoExecutor(
            folder=f"{args.output_dir}/{args.job_name}/submitit_logs",
            slurm_max_num_timeout=100,
            cluster=None if args.env.slurm else "debug",
        )
        additional_parameters = {}
        if args.env.nodelist != "":
            additional_parameters.update({"nodelist": args.env.nodelist})
        if args.env.exclude != "":
            additional_parameters.update({"exclude": args.env.exclude})
        executor.update_parameters(
            timeout_min=args.env.slurm_timeout,
            slurm_partition=args.env.slurm_partition,
            cpus_per_task=args.env.workers,
            gpus_per_node=args.env.ngpu,
            nodes=args.env.world_size,
            tasks_per_node=1,
            mem_gb=args.env.mem_gb,
            slurm_additional_parameters=additional_parameters,
            slurm_signal_delay_s=120)
        executor.update_parameters(name=slurm_job_name)
        executor.submit(Worker(), args)
    else:
        Worker()(args)


if __name__ == '__main__':
    main()
