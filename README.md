# DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring

Shaoqing Duan, Haofei Song, [Xintian Mao](https://scholar.google.es/citations?user=eM5Ogs8AAAAJ&hl=en),
Qingli Li, [Yan Wang](https://scholar.google.com/citations?user=5a1Cmk0AAAAJ&hl=en)

East China Normal University

Official implementation of **"Discontinuous Galerkin Neural Operator for Pathology
Defocus Deblurring"** (ICML 2026).

> Defocus deblurring in pathological microscopy remains challenging due to the spatially
> varying and locally discontinuous nature of optical blur induced by a position-dependent
> integral imaging process. Existing deep learning methods, constrained by shift-invariance
> assumptions and limited interpretability, are not well suited to such heterogeneous blur
> patterns. Neural operators provide a principled alternative by modeling defocus formation
> directly as an integral operator. However, most existing neural operator architectures
> rely on globally parameterized kernels that assume smoothness and stationarity, limiting
> their ability to model heterogeneous and locally discontinuous blur. To address this, we
> propose the **Discontinuous Galerkin Neural Operator (DGNO)**, which parameterizes the
> integral kernel using a discontinuous Galerkin formulation with element-local volume
> operators and interface numerical fluxes. DGNO provides a principled combination of
> locality, heterogeneity modeling, and global coherence while preserving the underlying
> physics of optical image formation.

## Model variants

DGNO comes in two operator variants (see paper), both produced by a single training
architecture **`DGNO`** (`basicsr/models/archs/dgno_arch.py`); the yml field
`network_g.variant` selects which discontinuous Galerkin operator is instantiated:

| Variant | `network_g.variant` | Description |
|---|---|---|
| **DGNO-Face** | `face` | Face-wise discontinuous Galerkin operator (`DG_Win_SRNO_FaceWise`) |
| **DGNO-Cell** | `cell` | Cell (element)-wise discontinuous Galerkin operator (`DG_Win_SRNO`) |

Inference uses a single unified evaluation architecture **`DGNO_eval`**
(`basicsr/models/archs/dgno_eval_arch.py`), into which both Face and Cell checkpoints load.

## Installation

```bash
conda create -n dgno python=3.9 -y
conda activate dgno

pip install -r requirements.txt
pip install mamba-ssm          # compiled CUDA extension; needs a matching torch/CUDA toolchain
python setup.py develop --no_cuda_ext
```

A CUDA GPU is required (`mamba-ssm`'s selective-scan kernels run on GPU).

## Datasets

The paper evaluates DGNO on three microscopy defocus-deblurring datasets:

- **BBBC006** ([Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC006)) —
  fluorescence microscopy in two channels: `w1` (Hoechst-stained nuclei) and `w2`
  (Phalloidin-stained actin). Single-channel grayscale, 696×520. Images at the optimal
  focal plane (z-stack = 16) are the sharp ground truth; defocused planes (z-stack =
  [2, 6, 10]) are the blurry inputs. 6,144 pairs, 4:1 train/test split. The preprocessed
  copy used in this work is mirrored on Hugging Face as `BBBC006.tar.gz` at
  [Duane245/DGNO](https://huggingface.co/Duane245/DGNO); extracting it yields the
  `BBBC006/{train,test}/{GT,blur}/*.tif` layout below directly.
- **3DHistech** ([Geng et al., *Cervical cytopathology image refocusing via multi-scale
  attention features and domain normalization*, Medical Image Analysis
  2022](https://doi.org/10.1016/j.media.2022.102566)) — cytopathology dataset captured with
  a 3DHistech digital scanner across multiple focal planes; the in-focus image is the sharp
  ground truth. RGB, 94,973 patches of 256×256, split 66,976 / 9,088 / 18,909 for
  train / val / test. Released by the original authors at
  [ShenghuaCheng/Cytopathology-image-refocusing](https://github.com/ShenghuaCheng/Cytopathology-image-refocusing);
  download from [Baidu Netdisk](https://pan.baidu.com/s/11VsxE8n_uX4G2Nn2jnWWwg)
  (access code: `refo`).

Place (or symlink) the data under `datasets/`:

```
datasets/
  BBBC006/
    train/{GT,blur}/*.tif
    test/{GT,blur}/*.tif
  3DHistech/
    train/{sharp,blur}/*.png
    val/{sharp,blur}/*.png
    test/{sharp,blur}/*.png
```

For BBBC006, download the preprocessed tarball (~3.2 GB, both `w1` and `w2` channels)
from Hugging Face and extract it under `datasets/`:

```bash
pip install -U huggingface_hub
huggingface-cli download Duane245/DGNO BBBC006.tar.gz --local-dir datasets
tar -xzf datasets/BBBC006.tar.gz -C datasets && rm datasets/BBBC006.tar.gz
```

This yields `datasets/BBBC006/{train,test}/{GT,blur}/*.tif` directly — no further
preprocessing needed. A single download covers both channels; the `w1` / `w2` split is
selected at run time from the yml `datasets.*.BBBCw` field.

## Pretrained checkpoints

Six checkpoints (3 datasets × 2 variants) are hosted on Hugging Face:
**https://huggingface.co/Duane245/DGNO**

| Dataset | DGNO-Face | DGNO-Cell |
|---|---|---|
| BBBC006 w1 | `DGNO_BBBC006_w1_Face.pth` | `DGNO_BBBC006_w1_Cell.pth` |
| BBBC006 w2 | `DGNO_BBBC006_w2_Face.pth` | `DGNO_BBBC006_w2_Cell.pth` |
| 3DHistech  | `DGNO_3DHistech_Face.pth`  | `DGNO_3DHistech_Cell.pth`  |

The checkpoints are stored under `pretrained/` inside the HF repo, so download into the
repo root (`--local-dir .`) and they land in `pretrained/`:

```bash
pip install -U huggingface_hub

# all six checkpoints (the --include filter skips the BBBC006.tar.gz dataset)
huggingface-cli download Duane245/DGNO --include "pretrained/*" --local-dir .

# or a single checkpoint — e.g. 3DHistech / Face
huggingface-cli download Duane245/DGNO pretrained/DGNO_3DHistech_Face.pth --local-dir .
```

## Training

Distributed training on 4 GPUs (`basicsr/train.py` under `torch.distributed.run`):

```bash
bash scripts/train_gpu0-3.sh options/DGNO_BBBC006_w1_Face.yml   # GPUs 0-3
bash scripts/train_gpu4-7.sh options/DGNO_BBBC006_w1_Cell.yml   # GPUs 4-7
```

Available configs in `options/`: `DGNO_BBBC006_w1_{Face,Cell}.yml`,
`DGNO_BBBC006_w2_{Face,Cell}.yml`, `DGNO_3DHistech_{Face,Cell}.yml`.

## Testing

```bash
# BBBC006 (w1 / w2) — channel is read from the yml
python test/test_bbbc.py \
    --config  options/DGNO_BBBC006_w1_Face.yml \
    --weights pretrained/DGNO_BBBC006_w1_Face.pth \
    --input_dir ./datasets/BBBC006/test --device cuda:0

# 3DHistech (RGB)
python test/test_3dhistech.py \
    --config  options/DGNO_3DHistech_Face.yml \
    --weights pretrained/DGNO_3DHistech_Face.pth \
    --input_dir ./datasets/3DHistech/test --device cuda:0
```

Add `--save_images` to write restored images to `results/`. Each script reports
PSNR / SSIM / LPIPS.

## Repository structure

```
basicsr/            trimmed BasicSR framework (training engine, data, metrics, utils)
  models/archs/     dgno_arch.py (training; variant=face|cell)  dgno_eval_arch.py
options/            6 training/eval configs (3 datasets × Face/Cell)
scripts/            distributed training launchers
test/               test_bbbc.py  test_3dhistech.py
pretrained/         place downloaded checkpoints here
datasets/           place datasets here
```

## Acknowledgements

This codebase is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and
[Restormer](https://github.com/swz30/Restormer). We thank the authors for their work.

## Citation

```bibtex
@inproceedings{duan2026dgno,
  title         = {Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring},
  author        = {Duan, Shaoqing and Song, Haofei and Mao, Xintian and Li, Qingli and Wang, Yan},
  booktitle     = {International Conference on Machine Learning (ICML)},
  year          = {2026},
  eprint        = {2605.23282},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url           = {https://arxiv.org/abs/2605.23282}
}
```

Preprint: [arXiv:2605.23282](https://arxiv.org/abs/2605.23282).
