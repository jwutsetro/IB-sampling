"""Dataset utilities for IB-sampling."""

from __future__ import annotations

import glob
import os
import random
import time
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import RandSpatialCropd, SpatialPadd


class MedicalPatchDataset(Dataset):
    """Dataset of pre-extracted 3D patches for medical images."""

    def __init__(
        self,
        root_dir: str,
        patch_size: tuple[int, int, int],
        transform: Optional[callable] = None,
        rank: int = 0,
        world_size: int = 1,
        patient_ids: Optional[List[str]] = None,
        is_training: bool = True,
        modalities: Optional[List[str]] = None,
    ) -> None:
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.is_training = is_training

        # Initialize shuffle and epoch attributes
        self.shuffle = True
        self.epoch = 0

        self.samples: List[Dict] = []
        self.modalities = modalities or self._discover_modalities(root_dir)

        lesion_patch_dir = os.path.join(root_dir, "lesion_patches")
        background_patch_dir = os.path.join(root_dir, "background_patches")
        self._load_samples(lesion_patch_dir, "positive", patient_ids)

        if self.is_training:
            self._load_samples(background_patch_dir, "negative", patient_ids)

        self.positive_indices = [i for i, s in enumerate(self.samples) if s["type"] == "positive"]
        self.negative_indices = [i for i, s in enumerate(self.samples) if s["type"] == "negative"]

        if self.is_training:
            self.split_positive_indices()
            self.positive_cache: Dict[int, Dict] = {}
            self.negative_cache: Dict[int, Dict] = {}
            self._preload_positive_samples()
        else:
            self.cache: Dict[int, Dict] = {}
            self._preload_all_samples()

    def split_positive_indices(self) -> None:
        total_positives = len(self.positive_indices)
        indices_per_gpu = total_positives // self.world_size

        indices = self.positive_indices.copy()
        if self.shuffle:
            random.seed(self.rank)
            random.shuffle(indices)

        indices = indices[: indices_per_gpu * self.world_size]
        start = self.rank * indices_per_gpu
        end = start + indices_per_gpu
        self.positive_indices = indices[start:end]
        self.num_positives = len(self.positive_indices)

    def _preload_positive_samples(self) -> None:
        print(f"GPU {self.rank}: Preloading positive samples into cache...")
        start_time = time.time()
        for idx in self.positive_indices:
            sample = self._load_sample(idx, apply_random_crop=False)
            self.positive_cache[idx] = sample
        end_time = time.time()
        print(
            f"GPU {self.rank}: Preloaded {len(self.positive_cache)} positive samples in {end_time - start_time:.2f} seconds."
        )

    def _preload_negative_samples(self, negative_indices: List[int]) -> None:
        print(f"GPU {self.rank}: Preloading negative samples into cache...")
        start_time = time.time()
        self.negative_cache.clear()
        for idx in negative_indices:
            sample = self._load_sample(idx, apply_random_crop=False)
            self.negative_cache[idx] = sample
        end_time = time.time()
        print(
            f"GPU {self.rank}: Preloaded {len(self.negative_cache)} negative samples in {end_time - start_time:.2f} seconds."
        )

    def _preload_all_samples(self) -> None:
        print(f"GPU {self.rank}: Preloading all validation samples into cache...")
        start_time = time.time()
        for idx in range(len(self.samples)):
            sample = self._load_sample(idx, apply_random_crop=False)
            self.cache[idx] = sample
        end_time = time.time()
        print(
            f"GPU {self.rank}: Preloaded {len(self.cache)} validation samples in {end_time - start_time:.2f} seconds."
        )

    def _discover_modalities(self, root_dir: str) -> List[str]:
        """Infer modality names from patch filenames."""
        modalities = set()
        for patch_dir in ["lesion_patches", "background_patches"]:
            path = os.path.join(root_dir, patch_dir)
            for fname in glob.glob(os.path.join(path, "*.nii*")):
                base = os.path.basename(fname)
                if "label" in base:
                    continue
                parts = base.replace(".nii.gz", "").replace(".nii", "").split("_")
                if len(parts) >= 3:
                    modalities.add(parts[-3])
        return sorted(modalities)

    def _load_sample(self, idx: int, apply_random_crop: bool = False) -> Dict[str, np.ndarray]:
        sample_info = self.samples[idx]
        modalities = sample_info["modalities"]
        label_path = sample_info["label_path"]

        modality_images = []
        for modality in self.modalities:
            image_path = modalities[modality]
            image = nib.load(image_path).get_fdata(dtype=np.float32)
            modality_images.append(image)

        image = np.stack(modality_images, axis=0)
        label = nib.load(label_path).get_fdata(dtype=np.float32)
        label = np.expand_dims(label, axis=0)

        sample = {"image": image, "label": label}

        pad_transform = SpatialPadd(keys=["image", "label"], spatial_size=self.patch_size, method="end")
        sample = pad_transform(sample)

        return sample

    def __len__(self) -> int:
        if self.is_training:
            return len(self.positive_cache) + len(self.negative_cache)
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.is_training:
            if idx in self.positive_cache:
                sample = self.positive_cache[idx]
                sample_type = "positive"
            elif idx in self.negative_cache:
                sample = self.negative_cache[idx]
                sample_type = "negative"
            else:
                sample = self._load_sample(idx, apply_random_crop=False)
                sample_type = self.samples[idx]["type"]
                if sample_type == "positive":
                    self.positive_cache[idx] = sample
                elif sample_type == "negative":
                    self.negative_cache[idx] = sample
                else:
                    raise ValueError(f"Unknown sample type {sample_type} for index {idx}")
            sample = self._apply_transforms(sample, sample_type=sample_type)
            return sample

        if idx in self.cache:
            sample = self.cache[idx]
        else:
            sample = self._load_sample(idx, apply_random_crop=False)
            self.cache[idx] = sample
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _apply_transforms(self, sample: Dict[str, np.ndarray], sample_type: str):
        if sample_type == "positive":
            crop = RandSpatialCropd(keys=["image", "label"], roi_size=self.patch_size, random_size=False)
            sample = crop(sample)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def _load_samples(self, patch_dir: str, sample_type: str, patient_ids: Optional[List[str]] = None) -> None:
        image_files = glob.glob(os.path.join(patch_dir, "*.nii*"))
        sample_dict: Dict[str, Dict] = {}
        for image_path in image_files:
            filename = os.path.basename(image_path)
            if "label" in filename:
                continue
            filename_no_ext = filename.replace(".nii.gz", "").replace(".nii", "")
            parts = filename_no_ext.split("_")
            if len(parts) < 4:
                continue
            patch_id = parts[-1]
            patch_type = parts[-2]
            modality = parts[-3]
            pid = "_".join(parts[:-3])

            if patient_ids is not None and pid not in patient_ids:
                continue

            patch_key = f"{pid}_{patch_type}_{patch_id}"
            if patch_key not in sample_dict:
                sample_dict[patch_key] = {
                    "patient_id": pid,
                    "patch_type": patch_type,
                    "patch_id": patch_id,
                    "modalities": {},
                    "label_path": "",
                    "type": sample_type,
                }
            sample_dict[patch_key]["modalities"][modality] = image_path

        for patch_key, sample_info in sample_dict.items():
            label_filename = (
                f"{sample_info['patient_id']}_label_{sample_info['patch_type']}_{sample_info['patch_id']}.nii.gz"
            )
            label_path = os.path.join(patch_dir, label_filename)
            if not os.path.exists(label_path):
                label_filename = label_filename.replace(".nii.gz", ".nii")
                label_path = os.path.join(patch_dir, label_filename)
                if not os.path.exists(label_path):
                    continue
            sample_info["label_path"] = label_path
            if set(self.modalities).issubset(sample_info["modalities"].keys()):
                self.samples.append(sample_info)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
