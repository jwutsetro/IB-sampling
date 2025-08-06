# IB-sampling

A modular dataloader for patch-based whole-body lesion detection. The package is
split into small components so the dataset, sampler and loader can be reused in
other projects or adapted to different modalities.

This repository accompanies the DEMI 2025 workshop paper:

**Instance‑Balanced Patch Sampling for Whole‑Body Lesion Segmentation**<br>
Joris Wuts, Jakub Ceranka, Jef Vandemeulebroucke, Frédéric Lecouvet<br>
_Open-access version and citation details will be added after the MICCAI 2025 conference._

Whole-body scans often contain many tiny lesions that make up less than 0.01% of the image volume.
Conventional positive–negative patch sampling struggles in this setting, over-representing background and large lesions while missing small targets.
The instance-balanced strategy samples patches per lesion instance, improving CPU data-loading efficiency, training speed and segmentation accuracy.

## Package structure

``ib_sampling`` exposes a few key utilities:

- `MedicalPatchDataset` – loads pre-extracted 3D patches and keeps a cache of
  frequently used patches in memory.
- `BalancedBatchSampler` – draws a balanced mix of positive and negative
  patches for each epoch.
- `get_loader` – convenience helper that builds training and validation
  dataloaders.
- `train_validate_dicts` – splits patient IDs into train/validation folds with
  a configurable number of splits via ``max_splits``.

The repository also provides a template ``prepare_dataset.py`` script that
converts raw volumes into patch datasets compatible with the dataloader.

## Raw data layout

Before running the preparation script, the raw data should be organised with one
directory per acquisition:

```
raw_dataset/
├── Patient01_a/
│   ├── T1.nii.gz
│   ├── b1000.nii.gz
│   └── GT.nii.gz
├── Patient01_b/
│   └── ...
└── Patient02_a/
    └── ...
```

Each folder contains one NIfTI volume per modality (any names are accepted) and
a ground-truth mask named ``GT.nii.gz``.

## Preparing patches

Use the provided script to extract positive and negative patches:

```
python prepare_dataset.py RAW_DIR OUTPUT_DIR --patch-size 128 128 128 \
       --modalities T1 b1000
```

The script performs 1–99 percentile normalisation on every modality, crops
positive patches around each connected component with a 10‑voxel margin and a
minimum size of ``128³``, and extracts background patches with a sliding window
of the same size and 50 % overlap. Patches are written to
``OUTPUT_DIR/lesion_patches`` and ``OUTPUT_DIR/background_patches`` using the
following convention:

- ``<pid>_<mod>_positive_<idx>.nii.gz`` and
  ``<pid>_label_positive_<idx>.nii.gz``
- ``<pid>_<mod>_negative_<idx>.nii.gz`` and
  ``<pid>_label_negative_<idx>.nii.gz``

## Using the dataloader

```python
from argparse import Namespace
from ib_sampling.loader import get_loader

args = Namespace(
    data_dir="OUTPUT_DIR",
    roi_x=128, roi_y=128, roi_z=128,
    batch_size=4,
    ratio=1.0,                 # negative:positive ratio
    split=1, max_splits=5,
    seed=0, rank=0, world_size=1,
    num_workers=4, distributed=False,
    modalities=["T1", "b1000"],
)

train_loader, val_loader = get_loader(args)
```

Each item returned by the loaders is a dictionary with ``image`` and ``label``
keys containing tensors shaped ``(C, Z, Y, X)`` and ``(1, Z, Y, X)``
respectively.

## Dataloader mechanics

- When instantiated, ``MedicalPatchDataset`` discovers the available modalities
  and preloads all positive patches into memory. In a distributed setup the
  positives are evenly split across workers.
- For every epoch, ``BalancedBatchSampler`` randomly selects the required number
  of negative patches to satisfy the desired ratio. The dataset then preloads
  only those negatives into a cache before iteration begins.
- The sampler reports the number of positives and negatives each epoch and
  yields a balanced list of indices. The dataloader fetches the cached samples
  and applies the requested transforms.
- During validation all patches are cached up-front, removing disk I/O during
  evaluation.

This design keeps GPU utilisation high by avoiding repeated disk reads while
still supporting large background pools.

## Authors

- Joris Wuts
- Jakub Ceranka
- Jef Vandemeulebroucke
- Frédéric Lecouvet

