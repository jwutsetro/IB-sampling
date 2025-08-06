"""Custom samplers used in IB-sampling."""

from __future__ import annotations

from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[int]):
    """Sampler that yields a balanced mix of positive and negative patches."""

    def __init__(
        self,
        dataset,
        ratio: float = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.ratio = ratio
        self.shuffle = shuffle
        self.epoch = 0

        if num_replicas is None:
            self.num_replicas = (
                torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            )
        else:
            self.num_replicas = num_replicas

        if rank is None:
            self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        else:
            self.rank = rank

        self.positive_indices = self.dataset.positive_indices
        self.num_positives = len(self.positive_indices)
        self.num_negatives_per_epoch = int(self.num_positives * self.ratio)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank)
        self.positive_indices = self.dataset.positive_indices
        self.num_positives = len(self.positive_indices)
        self.num_negatives_per_epoch = int(self.num_positives * self.ratio)

        negative_indices_all = self.dataset.negative_indices
        num_negatives = self.num_negatives_per_epoch
        negative_indices: List[int] = []
        if len(negative_indices_all) >= num_negatives:
            indices = torch.randperm(len(negative_indices_all), generator=g)[:num_negatives].tolist()
            negative_indices = [negative_indices_all[i] for i in indices]
        else:
            negative_indices = negative_indices_all.copy()
            extra_needed = num_negatives - len(negative_indices)
            if extra_needed > 0:
                indices = torch.randint(len(negative_indices_all), size=(extra_needed,), generator=g).tolist()
                negative_indices += [negative_indices_all[i] for i in indices]

        all_indices = self.positive_indices + negative_indices

        if self.shuffle:
            indices = torch.randperm(len(all_indices), generator=g).tolist()
            all_indices = [all_indices[i] for i in indices]

        self.dataset._preload_negative_samples(negative_indices)
        self.num_samples = len(all_indices)

        print(
            f"GPU {self.rank}: Number of positives: {len(self.positive_indices)}, Number of negatives: {len(negative_indices)}"
        )

        return iter(all_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
