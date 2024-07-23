from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from util import distributed as dist_utils


def get_dataloader(db, distributed, batch_size, workers, collate_fn=default_collate, shuffle=True, drop_last=True):
    if distributed:
        num_tasks = dist_utils.get_world_size()
        global_rank = dist_utils.get_rank()
        sampler = data.DistributedSampler(db, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
    else:
        sampler = data.RandomSampler(db, replacement=True)
    return data.DataLoader(
        db,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=workers>0,
    )