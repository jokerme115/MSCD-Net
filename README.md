# MSCD-Net: Cassava Rot Segmentation and Quantification (Paper Reproduction Code)

English | [简体中文](README.zh-CN.md)

This repository contains the experimental code for a research paper. It implements a DeepLabV3+-based binary semantic segmentation framework for cassava rot region detection, and provides a quantification pipeline (PPD) based on predicted masks. The code is configuration-driven and supports batched comparisons across backbones, loss functions, and oversampling strategies for ablation and reproducibility.

## 1. Method Overview

### 1.1 Task Definition

- Input: RGB images of cassava tubers
- Output: pixel-wise binary segmentation mask (0: healthy/background, 1: rotten region)

### 1.2 Network Architecture

This project is built on DeepLabV3+ and supports multiple backbones (ResNet / MobileNetV2 / Xception / HRNetV2). To strengthen feature representation for rot regions, an optional CBAM attention module is provided (channel/spatial re-weighting on ASPP outputs or the concatenated V3+ decoder features).

### 1.3 Class Imbalance Handling

Rot regions are typically sparse. The training pipeline supports oversampling for images that contain rot:

- Default repetition-based oversampling (default)
- SMOTE-family methods (smote / smoteenn / smotetomek, etc.): enabled when `imbalanced-learn` is installed; otherwise it falls back to default repetition.

### 1.4 Severity Quantification (PPD)

During inference, the model predicts a rot mask and uses OpenCV to estimate the cassava-region mask from the original image, then computes:

- PPD = predicted rotten area (restricted within cassava region) / cassava region area

The script exports side-by-side visualizations, PPD analysis overlays, and summary tables.

## 2. Code Structure

```
MSCD-Net/
├─ main.py                      # Training entry (config-driven; batched experiments)
├─ predict.py                   # Inference + PPD computation, visualization, and summaries
├─ config.yaml                  # Experiment config (backbone/loss/oversampling/training settings)
├─ datasets/
│  ├─ cassava_rot.py            # CassavaRotDataset (oversampling + label mapping)
│  └─ __init__.py
├─ network/                     # DeepLabV3/DeepLabV3+ and backbone implementations
├─ utils/                       # Training, augmentation, loss, scheduler, logging, etc.
└─ metrics/                     # Segmentation metrics (IoU/Dice/Recall/Precision/FP Rate, etc.)
```

## 3. Dataset Organization

The dataset root is specified by `data_root` in `config.yaml`. `CassavaRotDataset` expects:

```
datasets/data/
├─ train_aug.txt
├─ val.txt
├─ images/
│  ├─ train/
│  │  ├─ xxx.jpg
│  │  └─ ...
│  └─ val/
│     ├─ xxx.jpg
│     └─ ...
└─ labels_png/
   ├─ train/
   │  ├─ xxx.png
   │  └─ ...
   └─ val/
      ├─ xxx.png
      └─ ...
```

Label convention (mapped automatically during training):

- Pixel value `255` in label PNG is treated as rot and mapped to class `1`
- All other pixels are treated as healthy/background, class `0`

## 4. Environment and Dependencies

This project is a Python/PyTorch training and inference codebase. It does not ship a `requirements.txt`. Based on the actual imports, the commonly required packages include:

- PyTorch, torchvision
- numpy, tqdm, PyYAML
- Pillow (image I/O and visualization composition)
- matplotlib (visualization-related utilities)
- opencv-python (`predict.py` estimates cassava region mask)
- pandas (`predict.py` writes PPD summary tables)
- imbalanced-learn (optional; SMOTE-family oversampling)

## 5. Training (Batched Comparative Experiments)

Training is driven by `main.py`, which loads the YAML config and runs batched combinations.

```bash
python main.py --config_file config.yaml
```

The attention switch can be set in the config or overridden via CLI:

```bash
python main.py --config_file config.yaml --use_attention
```

### 5.1 Key Configuration Fields (config.yaml)

- `backbones`: list of backbones to compare
- `loss_functions`: list of loss functions to compare
- `oversample`: enable/disable oversampling
- `oversample_methods`: list of oversampling methods to compare
- `total_itrs` / `val_interval` / `print_interval`: training/validation schedule
- `lr` / `lr_policy` / `warmup_steps` / `warmup_lr`: learning rate strategy
- `crop_size` / `batch_size`: input crop size and batch size
- `experiment_name`: experiment root directory name

### 5.2 Outputs and Logs

Training artifacts are written to:

```
experiments/{experiment_name}/
├─ {experiment_name}_{backbone}_{loss_function}[_oversample_xxx][_attention]/
│  ├─ latest_*.pth
│  ├─ best_*.pth
│  ├─ early_stop_best.pth
│  ├─ training_log.csv
│  ├─ validation_log.csv
│  └─ logs/
├─ experiment_summary.csv
└─ detailed_experiment_summary.csv
```

## 6. Inference and PPD Computation

The inference script is `predict.py`, which reads the input and output paths from `config.yaml`:

- `predict_input`: a single image path or a directory of images
- `save_val_results_to`: output directory for visualizations and tables

Run:

```bash
python predict.py
```

### 6.1 Checkpoint Path Convention

`predict.py` currently loads the model checkpoint from:

```
models/best_deeplabv3plus_resnet101_cassava_rot_os16.pth
```

If your best checkpoint is under `experiments/`, copy/rename it to this path or edit `ckpt_path` in `predict.py`.

### 6.2 Inference Outputs

In `save_val_results_to`, the script writes:

- `*_comparison.png`: original image and predicted mask side-by-side (with PPD text)
- `*_ppd_analysis.png`: PPD overlay (red: cassava region, green: rotten region)
- `ppd_summary.csv`: per-image PPD values
- `ppd_summary_stats.csv`: aggregate statistics (mean/min/max, etc.)

## 7. Reproducibility Notes

For paper-level reproducibility, the training entry sets random seeds after loading the config. For consistent results, keep:

- the same `config.yaml` (especially `crop_size`, `batch_size`, `lr_policy`, `total_itrs`)
- the same split files (`train_aug.txt`, `val.txt`)
- the same backbone and loss combinations

## 8. Acknowledgements

This project is based on a classic DeepLabV3+ implementation and has been adapted for paper experiments, including batched comparisons, optional attention, oversampling strategies, and the PPD quantification pipeline.

