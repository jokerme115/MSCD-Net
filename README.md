# DeepLabV3Plus-Pytorch for Cassava Rot Detection

This repository is modified to support cassava rot detection task with binary segmentation.

## 主要特性

- 支持多种DeepLabV3+骨干网络（ResNet50, ResNet101, Xception等）
- 多种损失函数（Focal Loss, Dice Loss, Tversky Loss, Combined Loss）
- 数据增强策略
- 可视化工具

## 配置文件

项目支持通过JSON配置文件来设置训练参数。配置文件示例：

```json
{
    // 数据集相关配置
    "data_root": "./datasets/data",                 // 数据集根目录
    "dataset": "cassava_rot",                       // 数据集名称
    "num_classes": 2,                               // 类别数量
    "crop_size": 384,                               // 裁剪尺寸，减小以节省内存
    "batch_size": 8,                                // 批次大小
    
    // 模型相关配置
    "model": "deeplabv3plus_resnet50",              // 模型名称
    "separable_conv": false,                        // 是否使用可分离卷积
    "output_stride": 16,                            // 输出步长
    
    // 训练相关配置
    "total_itrs": 30000,                            // 总迭代次数
    "val_interval": 1000,                           // 验证间隔
    "lr": 0.01,                                     // 学习率
    "gpu_id": "0",                                  // GPU ID
    "lr_policy": "cosine",                          // 学习率调度策略
    "warmup_steps": 500,                            // warmup步数
    "warmup_lr": 1e-6,                              // warmup初始学习率
    "freeze_backbone": 0,                           // 冻结骨干网络的epoch数
    
    // 损失函数配置
    "loss_type": "combined",                        // 损失函数类型
    
    // 实验相关配置
    "experiment_name": "experiment_1",              // 实验名称，用于保存结果和模型
    "verbose": false                                // 是否显示详细信息
}
```

## Dataset Structure

For cassava rot detection, you need to organize your dataset in the following structure:

```
datasets/
└── data/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── val/
        ├── images/
        └── masks/
```

Each label mask should be a binary image where:
- 0 represents non-rotten areas
- 1 represents rotten areas

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

项目现在采用纯配置文件驱动的方式进行训练，所有训练参数都通过JSON配置文件进行设置。

直接运行训练脚本即可开始训练，将自动使用默认的[config.json](file://c:\Users\Administrator\Desktop\DeepLabV3Plus-Pytorch-master\config.json)配置文件：

```bash
python comprehensive_test.py
```

要使用不同的配置文件，只需修改[comprehensive_test.py](file://c:\Users\Administrator\Desktop\DeepLabV3Plus-Pytorch-master\comprehensive_test.py)中的配置文件路径：

```python
# 默认使用config.json配置文件
config_file = 'your_config.json'  # 修改为你的配置文件路径
opts = load_config_from_json(config_file)
train_and_evaluate(opts)
```

所有训练参数（包括模型架构、损失函数类型、数据增强策略等）都在配置文件中定义，无需通过命令行参数设置。

## 实验结果

每次实验的结果（模型、日志、评估结果）都会保存在`experiments/[experiment_name]`目录下，其中`experiment_name`在配置文件中定义。

## Loss Functions

We support several loss functions for binary segmentation:
- `bce_with_logits`: Binary Cross Entropy with Logits
- `dice_loss`: Dice Loss
- `focal_loss`: Focal Loss

## 可视化掩码

要可视化数据集中生成的掩码和原始标签区域，可以运行以下命令：

```bash
python visualize_masks.py --num_samples 10
```

这将生成图像，显示：
1. 原始图像
2. 标签区域（从边界框转换而来）
3. 生成的掩码

结果将保存在 `mask_visualizations` 目录中。

## Data Augmentation and Balancing

### Enhanced Data Augmentation

To improve model generalization, we implement comprehensive data augmentation strategies:

1. **Random Scaling**: Images are randomly scaled between 0.5x to 2.0x
2. **Random Cropping**: Random crops of fixed size with padding if needed
3. **Horizontal Flipping**: Random horizontal flips with 50% probability
4. **Random Rotation**: Images rotated by up to 30 degrees in either direction
5. **Color Jittering**: Adjustments to brightness, contrast, saturation, and hue

### Balanced Dataset Sampling

To address class imbalance in cassava rot detection (where rotten areas are typically rare), we provide a balanced dataset sampling approach:

1. **Oversampling**: Rotten samples are oversampled during training
2. **Balanced Selection**: 50% of training batches contain samples with rotten regions
3. **Usage**: Enable with `--use_balanced_dataset` flag

Example usage:
```bash
python comprehensive_test.py --use_balanced_dataset
```

## Model Improvements

### Self-Attention Mechanism

We have added a self-attention mechanism to the DeepLabV3+ model to improve global context modeling. This mechanism helps the model focus on relevant regions of the image when making predictions, which is especially beneficial for detecting irregular shapes like cassava rot areas.

The self-attention module is integrated into the decoder part of the DeepLabV3+ architecture, right after feature concatenation and before the final classification layers.

### Multi-Head Attention Implementation

To further enhance the attention mechanism, we have implemented a multi-head attention approach that can capture dependencies at different scales and positions:

1. **Multiple Attention Heads**: Instead of computing a single attention map, we compute multiple attention maps in parallel, each focusing on different aspects of the features.
2. **Better Representation**: Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
3. **Improved Performance**: This approach can better capture complex patterns in cassava rot regions.

### Efficient Attention Implementation

To address the computational overhead of self-attention, we have implemented efficient attention mechanisms that significantly reduce computation while maintaining performance:

1. **Reduced Complexity**: Multi-head attention distributes the computation across multiple heads, reducing the dimensionality of each head.
2. **Maintained Performance**: Despite the reduced complexity, the model still captures important relationships effectively.

## Training Optimizations

### Advanced Learning Rate Scheduling

We support multiple learning rate scheduling policies to optimize the training process:

1. **Polynomial Decay** (default): Traditional polynomial decay used in DeepLab
2. **Step Decay**: Reduces learning rate at specific intervals
3. **Cosine Annealing**: Smoothly decreases learning rate following a cosine curve

Example usage:
```bash
python main.py --lr_policy cosine --total_itrs 30000
```

### Early Stopping

To prevent overfitting and reduce training time, we implement early stopping based on validation performance:

1. **Monitoring**: The training process monitors Mean IoU on the validation set
2. **Patience**: Training stops if no improvement is observed for a specified number of evaluations
3. **Model Saving**: Only the best model (based on validation IoU) is saved

## Post-Processing Techniques

### CRF (Conditional Random Fields)

We provide post-processing with DenseCRF to refine segmentation boundaries:

1. **Boundary Refinement**: CRF helps sharpen object boundaries and reduce noise
2. **Context Modeling**: Incorporates spatial and color continuity constraints
3. **Usage**: Available in the `predict_with_crf.py` script

Example usage:
```bash
python predict_with_crf.py --input datasets/data/images --ckpt checkpoints/best_model.pth --use_crf
```

## Model Outputs

For cassava rot detection, the model outputs 2 channels:
- Channel 0: Non-rotten areas
- Channel 1: Rotten areas

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- CUDA (optional but recommended)
- pydensecrf (optional, for CRF post-processing)

Install pydensecrf with:
```bash
pip install pydensecrf
```

## 可视化

使用`visualize_pipeline.py`进行数据增强和预测结果的可视化。

## 预测

使用`predict.py`对新图像进行预测。

## References

This code is based on the original DeepLabV3Plus implementation.
# DeepLabv3Plus-Pytorch

Pretrained DeepLabv3, DeepLabv3+ for Pascal VOC & Cityscapes.

## Quick Start 

### 1. Available Architectures
| DeepLabV3    |  DeepLabV3+        |
| :---: | :---:     |
|deeplabv3_resnet50|deeplabv3plus_resnet50|
|deeplabv3_resnet101|deeplabv3plus_resnet101|
|deeplabv3_mobilenet|deeplabv3plus_mobilenet ||
|deeplabv3_hrnetv2_48 | deeplabv3plus_hrnetv2_48 |
|deeplabv3_hrnetv2_32 | deeplabv3plus_hrnetv2_32 |
|deeplabv3_xception | deeplabv3plus_xception |

please refer to [network/modeling.py](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/modeling.py) for all model entries.

Download pretrained models: [Dropbox](https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5vy6HdIhmRCTKHSa?dl=0), [Tencent Weiyun](https://share.weiyun.com/qqx78Pv5)

Note: The HRNet backbone was contributed by @timothylimyl. A pre-trained backbone is available at [google drive](https://drive.google.com/file/d/1NxCK7Zgn5PmeS7W1jYLt5J9E0RRZ2oyF/view?usp=sharing).

### 2. Load the pretrained model:
```python
model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_SRTIDE)
model.load_state_dict( torch.load( PATH_TO_PTH )['model_state']  )
```
### 3. Visualize segmentation outputs:
```python
outputs = model(images)
preds = outputs.max(1)[1].detach().cpu().numpy()
colorized_preds = val_dst.decode_target(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
# Do whatever you like here with the colorized segmentation maps
colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
```

### 4. Atrous Separable Convolution

**Note**: All pre-trained models in this repo were trained without atrous separable convolution.

Atrous Separable Convolution is supported in this repo. We provide a simple tool ``network.convert_to_separable_conv`` to convert ``nn.Conv2d`` to ``AtrousSeparableConvolution``. **Please run main.py with '--separable_conv' if it is required**. See 'main.py' and 'network/_deeplab.py' for more details. 

### 5. Prediction
Single image:
```bash
python predict.py --input datasets/data/cityscapes/leftImg8bit/train/bremen/bremen_000000_000019_leftImg8bit.png  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results
```

Image folder:
```bash
python predict.py --input datasets/data/cityscapes/leftImg8bit/train/bremen  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results
```

### 6. New backbones

Please refer to [this commit (Xception)](https://github.com/VainF/DeepLabV3Plus-Pytorch/commit/c4b51e435e32b0deba5fc7c8ff106293df90590d) for more details about how to add new backbones.

### 7. New datasets

You can train deeplab models on your own datasets. Your ``torch.utils.data.Dataset`` should provide a decoding method that transforms your predictions to colorized images, just like the [VOC Dataset](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/bfe01d5fca5b6bb648e162d522eed1a9a8b324cb/datasets/voc.py#L156):
```python

class MyDataset(data.Dataset):
    ...
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
```


## Results

### 1. Performance on Pascal VOC2012 Aug (21 classes, 513 x 513)

Training: 513x513 random crop  
validation: 513x513 center crop

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        | Dropbox  | Tencent Weiyun  | 
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: | :----:   |
| DeepLabV3-MobileNet       | 16      |  6.0G      |   16/16  |  0.701     |    [Download](https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=0)       | [Download](https://share.weiyun.com/A4ubD1DD) |
| DeepLabV3-ResNet50         | 16      |  51.4G     |  16/16   |  0.769     |    [Download](https://www.dropbox.com/s/3eag5ojccwiexkq/best_deeplabv3_resnet50_voc_os16.pth?dl=0) | [Download](https://share.weiyun.com/33eLjnVL) |
| DeepLabV3-ResNet101         | 16      |  72.1G     |  16/16   |  0.773     |    [Download](https://www.dropbox.com/s/vtenndnsrnh4068/best_deeplabv3_resnet101_voc_os16.pth?dl=0)       | [Download](https://share.weiyun.com/iCkzATAw)  |
| DeepLabV3Plus-MobileNet   | 16      |  17.0G      |  16/16   |  0.711    |    [Download](https://www.dropbox.com/s/0idrhwz6opaj7q4/best_deeplabv3plus_mobilenet_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/djX6MDwM) |
| DeepLabV3Plus-ResNet50    | 16      |   62.7G     |  16/16   |  0.772     |    [Download](https://www.dropbox.com/s/dgxyd3jkyz24voa/best_deeplabv3plus_resnet50_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/uTM4i2jG) |
| DeepLabV3Plus-ResNet101     | 16      |  83.4G     |  16/16   |  0.783     |    [Download](https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/UNPZr3dk) |


### 2. Performance on Cityscapes (19 classes, 1024 x 2048)

Training: 768x768 random crop  
validation: 1024x2048

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        | Dropbox  |  Tencent Weiyun  |
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: |  :----:   |
| DeepLabV3Plus-MobileNet   | 16      |  135G      |  16/16   |  0.721  |    [Download](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) | [Download](https://share.weiyun.com/aSKjdpbL) 
| DeepLabV3Plus-ResNet101   | 16      |  N/A      |  16/16   |  0.762  |    [Download](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing) | N/A |


#### Segmentation Results on Pascal VOC2012 (DeepLabv3Plus-MobileNet)

<div>
<img src="samples/1_image.png"   width="20%">
<img src="samples/1_target.png"  width="20%">
<img src="samples/1_pred.png"    width="20%">
<img src="samples/1_overlay.png" width="20%">
</div>

<div>
<img src="samples/23_image.png"   width="20%">
<img src="samples/23_target.png"  width="20%">
<img src="samples/23_pred.png"    width="20%">
<img src="samples/23_overlay.png" width="20%">
</div>

<div>
<img src="samples/114_image.png"   width="20%">
<img src="samples/114_target.png"  width="20%">
<img src="samples/114_pred.png"    width="20%">
<img src="samples/114_overlay.png" width="20%">
</div>

#### Segmentation Results on Cityscapes (DeepLabv3Plus-MobileNet)

<div>
<img src="samples/city_1_target.png"   width="45%">
<img src="samples/city_1_overlay.png"  width="45%">
</div>

<div>
<img src="samples/city_6_target.png"   width="45%">
<img src="samples/city_6_overlay.png"  width="45%">
</div>


#### Visualization of training

![trainvis](samples/visdom-screenshoot.png)


## Pascal VOC

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

#### 2.1 Standard Pascal VOC
You can run train.py with "--download" option to download and extract pascal voc dataset. The defaut path is './datasets/data':

```
/datasets
    /data
        /VOCdevkit 
            /VOC2012 
                /SegmentationClass
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

#### 2.2  Pascal VOC trainaug (Recommended!!)

See chapter 4 of [2]

        The original dataset contains 1464 (train), 1449 (val), and 1456 (test) pixel-level annotated images. We augment the dataset by the extra annotations provided by [76], resulting in 10582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU).

*./datasets/data/train_aug.txt* includes the file names of 10582 trainaug images (val images are excluded). Please to download their labels from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those labels come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).

Extract trainaug labels (SegmentationClassAug) to the VOC2012 directory.

```
/datasets
    /data
        /VOCdevkit  
            /VOC2012
                /SegmentationClass
                /SegmentationClassAug  # <= the trainaug labels
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

### 3. Training on Pascal VOC2012 Aug

#### 3.1 Visualize training (Optional)

Start visdom sever for visualization. Please remove '--enable_vis' if visualization is not needed. 

```bash
# Run visdom server on port 28333
visdom -port 28333
```

#### 3.2 Training with OS=16

Run main.py with *"--year 2012_aug"* to train your model on Pascal VOC2012 Aug. You can also parallel your training on 4 GPUs with '--gpu_id 0,1,2,3'

**Note: There is no SyncBN in this repo, so training with *multple GPUs and small batch size* may degrades the performance. See [PyTorch-Encoding](https://hangzhang.org/PyTorch-Encoding/tutorials/syncbn.html) for more details about SyncBN**

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
```

#### 3.3 Continue training

Run main.py with '--continue_training' to restore the state_dict of optimizer and scheduler from YOUR_CKPT.

```bash
python main.py ... --ckpt YOUR_CKPT --continue_training
```

#### 3.4. Testing

Results will be saved at ./results.

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --test_only --save_val_results
```

## Cityscapes

### 1. Download cityscapes and extract it to 'datasets/data/cityscapes'

```
/datasets
    /data
        /cityscapes
            /gtFine
            /leftImg8bit
```

### 2. Train your model on Cityscapes

```bash
python main.py --model deeplabv3plus_mobilenet --dataset cityscapes --enable_vis --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 16 --output_stride 16 --data_root ./datasets/data/cityscapes 
```

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
