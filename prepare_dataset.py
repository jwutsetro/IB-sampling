"""Script to convert whole-body volumes into patch datasets.

For each subject folder containing multiple modalities (e.g., ``T1.nii.gz`` and
``b1000.nii.gz``) and a ground-truth mask ``GT.nii.gz``, this script extracts:

* **Positive patches**: one patch per connected component in the mask. Each
  patch uses the minimal bounding box around the lesion, padded by 10 voxels and
  enlarged to at least ``128^3`` (or a user-specified size).
* **Negative patches**: sliding window patches of ``128^3`` with 50% overlap over
  the volume, skipping any regions that intersect lesions.

All modalities are normalized using the 1st and 99th percentiles before patch
extraction. Patches are written to ``lesion_patches`` and ``background_patches``
subdirectories of the output folder using the naming convention expected by the
``MedicalPatchDataset`` loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def normalize_volume(img: np.ndarray) -> np.ndarray:
    """Clip to the 1st and 99th percentiles and scale to [0, 1]."""
    p1, p99 = np.percentile(img, (1, 99))
    if p99 <= p1:
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, p1, p99)
    return (img - p1) / (p99 - p1)


def ensure_size(start: int, end: int, min_size: int, max_dim: int) -> tuple[int, int]:
    """Expand ``start``/``end`` so the interval is at least ``min_size`` long."""
    size = end - start
    if size >= min_size:
        return start, end
    extra = min_size - size
    start = max(0, start - extra // 2)
    end = min(max_dim, end + extra - extra // 2)
    if end - start < min_size:
        if start == 0:
            end = min(max_dim, start + min_size)
        else:
            start = max(0, end - min_size)
    return start, end


def generate_indices(dim: int, patch: int, stride: int) -> list[int]:
    """Return start indices covering ``dim`` using ``patch`` size and ``stride``."""
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    if starts[-1] != dim - patch:
        starts.append(dim - patch)
    return starts


def save_patch(
    pid: str,
    patch_type: str,
    patch_idx: int,
    modalities: dict[str, np.ndarray],
    label: np.ndarray,
    out_dir: Path,
    affine: np.ndarray,
) -> None:
    for mod, data in modalities.items():
        out_path = out_dir / f"{pid}_{mod}_{patch_type}_{patch_idx:04d}.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine), out_path)
    label_path = out_dir / f"{pid}_label_{patch_type}_{patch_idx:04d}.nii.gz"
    nib.save(nib.Nifti1Image(label.astype(np.float32), affine), label_path)


def process_case(
    case_dir: Path,
    output_dir: Path,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    modalities_filter: list[str] | None,
) -> None:
    pid = case_dir.name
    lesion_dir = output_dir / "lesion_patches"
    background_dir = output_dir / "background_patches"
    lesion_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)

    modality_paths: dict[str, Path] = {}
    gt_path: Path | None = None
    for path in case_dir.glob("*.nii*"):
        name = path.name.lower()
        if name.startswith("gt"):
            gt_path = path
        else:
            mod = path.name.replace(".nii.gz", "").replace(".nii", "")
            if modalities_filter is None or mod in modalities_filter:
                modality_paths[mod] = path

    if gt_path is None or not modality_paths:
        return

    modalities = {}
    affine = None
    for mod, p in modality_paths.items():
        img = nib.load(p)
        data = normalize_volume(img.get_fdata(dtype=np.float32))
        modalities[mod] = data.astype(np.float32)
        if affine is None:
            affine = img.affine

    gt_img = nib.load(gt_path)
    label = gt_img.get_fdata(dtype=np.float32)
    if affine is None:
        affine = gt_img.affine

    # Positive patches
    labeled, num = ndi.label(label > 0)
    pos_idx = 0
    for comp in range(1, num + 1):
        mask = labeled == comp
        coords = np.where(mask)
        zmin, ymin, xmin = [int(c.min()) for c in coords]
        zmax, ymax, xmax = [int(c.max()) for c in coords]
        pad = 10
        zmin = max(zmin - pad, 0)
        ymin = max(ymin - pad, 0)
        xmin = max(xmin - pad, 0)
        zmax = min(zmax + pad + 1, label.shape[0])
        ymax = min(ymax + pad + 1, label.shape[1])
        xmax = min(xmax + pad + 1, label.shape[2])

        zmin, zmax = ensure_size(zmin, zmax, patch_size[0], label.shape[0])
        ymin, ymax = ensure_size(ymin, ymax, patch_size[1], label.shape[1])
        xmin, xmax = ensure_size(xmin, xmax, patch_size[2], label.shape[2])

        slices = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
        label_patch = label[slices]
        modality_patches = {m: img[slices] for m, img in modalities.items()}
        save_patch(pid, "positive", pos_idx, modality_patches, label_patch, lesion_dir, affine)
        pos_idx += 1

    # Negative patches
    neg_idx = 0
    z_starts = generate_indices(label.shape[0], patch_size[0], stride[0])
    y_starts = generate_indices(label.shape[1], patch_size[1], stride[1])
    x_starts = generate_indices(label.shape[2], patch_size[2], stride[2])
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                z_end, y_end, x_end = z + patch_size[0], y + patch_size[1], x + patch_size[2]
                slices = (slice(z, z_end), slice(y, y_end), slice(x, x_end))
                label_patch = label[slices]
                if np.any(label_patch > 0):
                    continue
                modality_patches = {m: img[slices] for m, img in modalities.items()}
                save_patch(pid, "negative", neg_idx, modality_patches, label_patch, background_dir, affine)
                neg_idx += 1

    print(f"{pid}: {pos_idx} positive patches, {neg_idx} negative patches")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset into patches")
    parser.add_argument("input_dir", help="Directory with raw subject folders")
    parser.add_argument("output_dir", help="Directory to store extracted patches")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument(
        "--stride",
        type=int,
        nargs=3,
        default=None,
        help="Stride for negative patch extraction (defaults to half patch size)",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of modalities to process (default: all found)",
    )
    args = parser.parse_args()

    patch_size = tuple(args.patch_size)
    if args.stride is None:
        stride = tuple(s // 2 for s in patch_size)
    else:
        stride = tuple(args.stride)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities_filter = args.modalities
    for case in sorted(input_dir.iterdir()):
        if case.is_dir():
            process_case(case, output_dir, patch_size, stride, modalities_filter)


if __name__ == "__main__":
    main()

