"""Dataloader construction utilities."""

from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data import list_data_collate
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ToTensord,
)

from .dataset import MedicalPatchDataset
from .sampler import BalancedBatchSampler
from .utils import train_validate_dicts


def get_loader(args):
    """Create training and validation data loaders."""
    root_dir = args.data_dir
    patch_size = (args.roi_x, args.roi_y, args.roi_z)
    batch_size = args.batch_size
    ratio = args.ratio
    modalities = getattr(args, "modalities", None)
    if isinstance(modalities, str):
        modalities = [m.strip() for m in modalities.split(",") if m.strip()]

    random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)

    train_transforms = Compose(
        [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_patient_ids, val_patient_ids = train_validate_dicts(root_dir, args)

    train_dataset = MedicalPatchDataset(
        root_dir,
        patch_size,
        transform=train_transforms,
        rank=args.rank,
        world_size=args.world_size,
        patient_ids=train_patient_ids,
        is_training=True,
        modalities=modalities,
    )

    if getattr(args, "distributed", False):
        train_sampler = BalancedBatchSampler(
            train_dataset,
            ratio=ratio,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
    else:
        train_sampler = BalancedBatchSampler(
            train_dataset,
            ratio=ratio,
            num_replicas=1,
            rank=0,
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    val_dataset = MedicalPatchDataset(
        root_dir,
        patch_size,
        transform=val_transforms,
        rank=args.rank,
        world_size=args.world_size,
        patient_ids=val_patient_ids,
        is_training=False,
        modalities=modalities,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    return train_loader, val_loader
