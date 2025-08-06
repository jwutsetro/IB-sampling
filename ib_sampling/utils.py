"""Utility helpers for data handling."""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np


def train_validate_dicts(data_dir: str, args) -> Tuple[List[str], List[str]]:
    """Split patient IDs into train and validation folds.

    Parameters
    ----------
    data_dir: str
        Directory containing ``lesion_patches`` and ``background_patches``.
    args: Namespace
        Requires ``split`` and ``max_splits`` attributes to control the number of
        cross-validation folds and which fold is used for validation.
    """
    lesion_patch_dir = os.path.join(data_dir, "lesion_patches")
    background_patch_dir = os.path.join(data_dir, "background_patches")

    lesion_files = glob.glob(os.path.join(lesion_patch_dir, "*.nii*"))
    background_files = glob.glob(os.path.join(background_patch_dir, "*.nii*"))

    all_files = lesion_files + background_files

    patient_ids = set()
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if "label" in filename:
            continue

        filename_no_ext = filename.replace(".nii.gz", "").replace(".nii", "")
        parts = filename_no_ext.split("_")

        if len(parts) < 4:
            continue

        patient_id = "_".join(parts[:-3])
        patient_ids.add(patient_id)

    patient_ids = sorted(list(patient_ids))
    print(f"All patient IDs: {patient_ids}")

    np.random.seed(42)
    np.random.shuffle(patient_ids)

    num_patients = len(patient_ids)
    max_splits = getattr(args, "max_splits", 7)
    num_val_patients = max(1, num_patients // max_splits)

    folds = [patient_ids[i : i + num_val_patients] for i in range(0, num_patients, num_val_patients)]
    if len(folds) > max_splits:
        folds[-2].extend(folds[-1])
        folds.pop()

    val_fold_index = (args.split - 1) % len(folds)
    val_patient_ids = folds[val_fold_index]
    train_patient_ids = [p for i, fold in enumerate(folds) if i != val_fold_index for p in fold]

    print(f"Training patient IDs: {train_patient_ids}")
    print(f"Validation patient IDs: {val_patient_ids}")

    return train_patient_ids, val_patient_ids
