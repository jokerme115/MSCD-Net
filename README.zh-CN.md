# MSCD-Net：木薯腐烂区域检测与量化（论文复现实验代码）

简体中文 | [English](README.md)

本仓库为论文实验代码，实现了以 DeepLabV3+ 为主体的二分类语义分割框架，用于木薯腐烂区域检测，并提供基于预测分割结果的腐烂程度量化指标（PPD）计算与可视化。代码以配置文件驱动，支持批量对比不同骨干网络、损失函数与过采样策略，便于完成消融实验与复现评估。

## 1. 方法概述

### 1.1 任务定义

- 输入：木薯块根 RGB 图像
- 输出：像素级二分类分割掩码（0：健康/背景；1：腐烂区域）

### 1.2 网络结构

本项目以 DeepLabV3+ 为基础实现，提供多种骨干网络（ResNet / MobileNetV2 / Xception / HRNetV2）。为增强对腐烂区域的表征能力，可选引入 CBAM 注意力模块（对 ASPP 输出或 V3+ 解码器拼接特征进行通道/空间注意力重标定）。

### 1.3 类别不平衡处理

腐烂区域通常占比很低，训练阶段支持对含腐烂样本进行过采样：

- 默认重复采样（default）
- SMOTE 系列方法（smote / smoteenn / smotetomek 等）：若安装 `imbalanced-learn` 则启用，否则自动回退为默认重复采样

### 1.4 腐烂程度量化（PPD）

推理阶段在预测腐烂掩码基础上，结合 OpenCV 在原图中估计“木薯区域”掩码，计算：

- PPD = 预测腐烂面积（限制在木薯区域内） / 木薯区域面积

并输出对比图、PPD 分析可视化图与汇总表格。

## 2. 代码结构

```
MSCD-Net/
├─ main.py                      # 训练入口（配置文件驱动；批量实验）
├─ predict.py                   # 推理与PPD计算、可视化与汇总
├─ config.yaml                  # 实验配置（骨干/损失/过采样/训练参数等）
├─ datasets/
│  ├─ cassava_rot.py            # CassavaRotDataset（支持过采样与标签映射）
│  └─ __init__.py
├─ network/                     # DeepLabV3/DeepLabV3+与骨干网络
├─ utils/                       # 训练、数据增强、损失函数、调度器、日志等
└─ metrics/                     # 分割评估指标（IoU/Dice/Recall/Precision/FP Rate等）
```

## 3. 数据组织

数据根目录由 `config.yaml` 的 `data_root` 指定，`CassavaRotDataset` 读取以下结构：

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

标签约定（训练时自动映射）：

- 原始标签图中像素值 `255` 视为腐烂区域，并映射为类别 `1`
- 其余像素视为健康/背景，类别 `0`

## 4. 环境与依赖

代码为 Python/PyTorch 训练与推理脚本集合，建议使用独立虚拟环境。项目未内置 `requirements.txt`，根据实际导入依赖，运行所需常用包如下：

- PyTorch、torchvision
- numpy、tqdm、PyYAML
- Pillow（用于图像读写与可视化合成）
- matplotlib（用于可视化相关功能）
- opencv-python（`predict.py` 使用 OpenCV 提取木薯区域）
- pandas（`predict.py` 输出 PPD 汇总表）
- imbalanced-learn（可选；用于 SMOTE 系列过采样方法）

## 5. 训练（批量对比实验）

训练由 `main.py` 驱动，读取 YAML 配置并执行批量组合实验。

```bash
python main.py --config_file config.yaml
```

注意力模块开关可通过配置文件或命令行覆盖：

```bash
python main.py --config_file config.yaml --use_attention
```

### 5.1 关键配置项（config.yaml）

- `backbones`：骨干网络列表（可多选以进行对比）
- `loss_functions`：损失函数列表（可多选以进行对比）
- `oversample`：是否启用过采样
- `oversample_methods`：过采样方法列表（可多选以进行对比）
- `total_itrs` / `val_interval` / `print_interval`：训练与验证节奏
- `lr` / `lr_policy` / `warmup_steps` / `warmup_lr`：学习率策略
- `crop_size` / `batch_size`：输入裁剪与批量大小
- `experiment_name`：实验根目录命名

### 5.2 结果与日志输出

训练输出位于：

```
experiments/{experiment_name}/
├─ {experiment_name}_{backbone}_{loss_function}[_oversample_xxx][_attention]/
│  ├─ latest_*.pth              # 周期性保存的检查点
│  ├─ best_*.pth                # 最优检查点（基于 Mean IoU）
│  ├─ early_stop_best.pth       # 早停保存的最优权重（若启用）
│  ├─ training_log.csv
│  ├─ validation_log.csv
│  └─ logs/                     # TensorBoard 目录（当前为占位写入）
├─ experiment_summary.csv
└─ detailed_experiment_summary.csv
```

## 6. 推理与 PPD 计算

推理脚本为 `predict.py`，从 `config.yaml` 读取输入路径与输出目录：

- `predict_input`：待预测的单张图像路径或图像目录
- `save_val_results_to`：可视化与表格输出目录

运行命令：

```bash
python predict.py
```

### 6.1 模型权重路径约定

`predict.py` 当前默认从以下路径加载模型权重：

```
models/best_deeplabv3plus_resnet101_cassava_rot_os16.pth
```

若训练得到的最优权重位于 `experiments/` 下，请将其复制/重命名到上述位置，或按需修改 `predict.py` 中的 `ckpt_path`。

### 6.2 推理输出

在 `save_val_results_to` 目录下生成：

- `*_comparison.png`：原图与分割结果并排对比（带 PPD 文本）
- `*_ppd_analysis.png`：PPD 分析图（红色：木薯区域；绿色：腐烂区域）
- `ppd_summary.csv`：逐图像 PPD 明细
- `ppd_summary_stats.csv`：PPD 统计汇总（均值/最小/最大等）

## 7. 复现说明

为保证论文结果可复现，本仓库训练入口在加载配置后会设置随机种子；建议在相同硬件环境与软件版本下复现实验，并保持：

- 相同的 `config.yaml`（尤其是 `crop_size`、`batch_size`、`lr_policy`、`total_itrs`）
- 相同的数据划分文件（`train_aug.txt`、`val.txt`）
- 相同的模型骨干与损失函数组合

## 8. 致谢

本项目模型实现参考了 DeepLabV3+ 的经典开源实现，并在此基础上针对论文任务进行了实验工程化改造（批量对比、注意力模块、过采样策略与 PPD 量化流程）。
