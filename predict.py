import torch
import torch.nn as nn
from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
from utils.config import load_config
import os
import numpy as np
import pandas as pd

from torch.utils import data
try:
    from datasets import CassavaRotDataset
except ImportError as e:
    raise ImportError("无法导入 CassavaRotDataset。请确保 datasets 模块在 PYTHONPATH 中。") from e
from torchvision import transforms as T
from metrics import StreamSegMetrics

from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import cv2
import yaml
import pandas as pd

def decode_target(mask):
    """将标签解码为可视化格式"""
    # 创建彩色标签图
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask == 1] = [0, 0, 255]  # 红色表示腐烂区域（类别1）
    color_mask[mask == 0] = [0, 0, 0]    # 黑色表示健康区域（类别0）
    return color_mask

def get_cassava_area_mask(image):
    """
    从原始图像中获取木薯区域mask
    使用OpenCV和多种颜色空间来更准确地确定木薯区域
    """
    # 将PIL图像转换为numpy数组，然后转换为BGR格式（OpenCV使用BGR）
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 转换到不同的颜色空间以更好地分离木薯区域
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    
    # 在不同颜色空间中定义木薯的颜色范围
    # HSV范围 (木薯通常呈黄色到棕色)
    lower_hsv = np.array([10, 30, 30])
    upper_hsv = np.array([40, 255, 255])
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    
    # Lab范围 (木薯在Lab空间中有特定的范围)
    lower_lab = np.array([120, 90, 120])
    upper_lab = np.array([240, 150, 180])
    mask_lab = cv2.inRange(img_lab, lower_lab, upper_lab)
    
    # YCrCb范围 (木薯在YCrCb空间中的范围)
    lower_ycrcb = np.array([70, 50, 120])
    upper_ycrcb = np.array([230, 140, 190])
    mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
    
    # 基于RGB的额外检测方法（作为备选方案）
    # 木薯通常呈黄色/棕色，R和G值较高，B值中等到较低
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r_channel = img_rgb[:,:,0]
    g_channel = img_rgb[:,:,1]
    b_channel = img_rgb[:,:,2]
    
    # 宽松的RGB阈值
    mask_rgb = ((r_channel > 40) & (r_channel < 255) &
                (g_channel > 40) & (g_channel < 255) &
                (b_channel > 20) & (b_channel < 230)).astype(np.uint8) * 255
    
    # 结合不同颜色空间的掩码
    combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)
    combined_mask = cv2.bitwise_or(combined_mask, mask_ycrcb)
    combined_mask = cv2.bitwise_or(combined_mask, mask_rgb)
    
    # 使用形态学操作清理掩码
    # 创建椭圆形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # 开运算去除小的噪声点
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
    
    # 闭运算填充区域内的小孔
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 膨胀操作使区域稍微扩大，确保包含完整的木薯
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # 查找轮廓并选择最大的几个轮廓（假设木薯是主要对象）
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建新的掩码只保留大的轮廓
    final_mask = np.zeros_like(combined_mask)
    if contours:
        # 根据轮廓面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 计算总面积阈值（至少占图像0.5%的面积）
        min_area = img_bgr.shape[0] * img_bgr.shape[1] * 0.005
        # 最多处理前10个最大的轮廓
        for i, contour in enumerate(contours[:10]):
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(final_mask, [contour], 255)
            elif i == 0:  # 如果最大的轮廓都不够大，但还是要保留它
                cv2.fillPoly(final_mask, [contour], 255)
    
    # 如果上述方法没有检测到足够大的区域，使用更宽松的方法
    if np.count_nonzero(final_mask) < (img_bgr.shape[0] * img_bgr.shape[1] * 0.01):
        # 使用更宽松的RGB阈值
        mask_relaxed = ((r_channel > 20) & (r_channel < 255) &
                       (g_channel > 20) & (g_channel < 255) &
                       (b_channel > 10) & (b_channel < 240)).astype(np.uint8) * 255
        
        # 简单的形态学操作
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_relaxed = cv2.morphologyEx(mask_relaxed, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        mask_relaxed = cv2.morphologyEx(mask_relaxed, cv2.MORPH_OPEN, small_kernel, iterations=1)
        
        # 查找并保留最大的轮廓
        contours, _ = cv2.findContours(mask_relaxed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 保留最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(final_mask, [largest_contour], 255)
    
    # 最后的形态学优化
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 转换为uint8类型并返回 (0和1值)
    result = (final_mask > 0).astype(np.uint8)
    
    return result

def draw_rotten_contours(image, pred_mask):
    """在图像上圈出腐烂区域轮廓"""
    # 将PIL图像转换为numpy数组
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # 将RGB转换为BGR（OpenCV使用BGR）
    if len(image_np.shape) == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_np
    
    # 将预测mask转换为uint8类型
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原图上绘制轮廓
    cv2.drawContours(image_cv, contours, -1, (0, 0, 255), 2)  # 红色轮廓，线宽为2
    
    # 转换回RGB
    if len(image_cv.shape) == 3:
        result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    else:
        result_image = image_cv
    
    return Image.fromarray(result_image)

def create_comparison_image(original_img, pred_mask, ppd):
    """创建对比图像，左侧为原始图像，右侧为预测结果"""
    # 获取原始图像尺寸
    width, height = original_img.size
    
    # 创建带腐烂区域标记的图像
    marked_img = draw_rotten_contours(original_img, pred_mask)
    
    # 创建分割结果图像
    colorized_preds = decode_target(pred_mask).astype('uint8')
    segmentation_img = Image.fromarray(colorized_preds)
    
    # 调整分割图像大小以匹配原始图像
    segmentation_img = segmentation_img.resize((width, height), Image.Resampling.LANCZOS)
    
    # 创建新的图像，宽度是原始图像的两倍
    combined_img = Image.new('RGB', (width * 2, height))
    
    # 粘贴原始图像和分割图像
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(segmentation_img, (width, 0))
    
    # 添加文字信息
    draw = ImageDraw.Draw(combined_img)
    text = f"PPD: {ppd:.4f}"
    draw.text((10, 10), text, fill=(255, 0, 0))  # 红色文字
    
    # 添加标签说明
    draw.text((10, height - 30), "Original", fill=(0, 255, 0))
    draw.text((width + 10, height - 30), "Prediction", fill=(0, 255, 0))
    
    return combined_img

def visualize_ppd_analysis(original_img, pred_mask, cassava_mask, ppd, save_path=None):
    """
    可视化PPD分析结果，用绿色透明显示腐烂区域（分子），用红色透明显示木薯区域（分母）
    """
    # 将PIL图像转换为numpy数组
    img_array = np.array(original_img)
    
    # 确保所有mask尺寸一致
    if cassava_mask.shape != pred_mask.shape:
        cassava_mask = Image.fromarray(cassava_mask)
        cassava_mask = cassava_mask.resize(pred_mask.shape[::-1], Image.NEAREST)
        cassava_mask = np.array(cassava_mask)
    
    # 创建可视化图像
    vis_img = img_array.copy()
    
    # 用红色半透明显示木薯区域（分母）
    red_overlay = np.zeros_like(img_array)
    red_overlay[:, :, 0] = 255  # 红色通道
    
    # 应用红色覆盖到木薯区域
    cassava_bool = cassava_mask.astype(bool)
    if len(cassava_bool.shape) > 2:
        cassava_bool = cassava_bool[:, :, 0]  # 如果是3通道，取第一个通道
    vis_img[cassava_bool] = (vis_img[cassava_bool] * 0.5 + red_overlay[cassava_bool] * 0.5).astype(np.uint8)
    
    # 用绿色半透明显示腐烂区域（分子）
    green_overlay = np.zeros_like(img_array)
    green_overlay[:, :, 1] = 255  # 绿色通道
    
    # 获取腐烂区域（预测结果中标签为1的区域）
    rotten_bool = (pred_mask == 1)
    vis_img[rotten_bool] = (vis_img[rotten_bool] * 0.5 + green_overlay[rotten_bool] * 0.5).astype(np.uint8)
    
    # 转换回PIL图像
    result_img = Image.fromarray(vis_img)
    
    # 添加文字信息
    draw = ImageDraw.Draw(result_img)
    text = f"PPD: {ppd:.4f}"
    draw.text((10, 10), text, fill=(255, 255, 255))  # 白色文字
    
    # 添加图例说明
    draw.text((10, 30), "Red: Cassava Area (Denominator)", fill=(255, 255, 255))
    draw.text((10, 50), "Green: Rotten Area (Numerator)", fill=(255, 255, 255))
    
    if save_path:
        result_img.save(save_path)
    
    return result_img

def main():
    # 加载配置文件
    config = load_config('config.yaml')
    
    # 使用配置文件中的参数
    dataset_name = config.get('dataset', 'cassava_rot')
    num_classes = config.get('num_classes', 2)
    model_name = 'deeplabv3plus_resnet101'  # 固定使用指定模型
    separable_conv = config.get('separable_conv', False)
    output_stride = config.get('output_stride', 16)
    crop_val = False
    crop_size = config.get('crop_size', 513)
    gpu_id = config.get('gpu_id', '0')
    input_path = config.get('predict_input', './datasets/data/images/val')  # 从配置文件获取输入路径或使用默认值
    data_root = config.get('data_root', './datasets/data')
    
    # 设置保存路径
    save_val_results_to = config.get('save_val_results_to', './predict_results')
    
    if dataset_name.lower() == 'cassava_rot':
        num_classes = 2

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(input_path):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(input_path, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(input_path):
        image_files.append(input_path)
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # Set up model (all models are 'constructed at network.modeling)
    # 根据配置文件决定是否启用注意力机制
    use_attention = config.get('use_attention', False)
    model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride, use_attention=use_attention)
    if separable_conv and 'plus' in model_name:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # 从models目录加载指定模型
    ckpt_path = os.path.join('models', 'best_deeplabv3plus_resnet101_cassava_rot_os16.pth')
    if os.path.isfile(ckpt_path):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt_path)
        del checkpoint
    else:
        print("[!] 没有找到模型文件 %s" % ckpt_path)
        return

    if crop_val:
        transform = T.Compose([
                T.Resize(crop_size),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    
    # 存储所有图片的PPD结果
    ppd_results = []
    
    if save_val_results_to is not None:
        os.makedirs(save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            try:
                ext = os.path.basename(img_path).split('.')[-1]
                img_name = os.path.basename(img_path)[:-len(ext)-1]
                original_img = Image.open(img_path).convert('RGB')
                img = transform(original_img).unsqueeze(0) # To tensor of NCHW
                img = img.to(device)
                
                pred = model(img).max(1)[1].cpu().numpy()[0] # HW
                
                # 使用OpenCV从原始图像中获取木薯区域
                cassava_mask = get_cassava_area_mask(original_img)
                # 调整大小以匹配预测结果
                cassava_mask_img = Image.fromarray(cassava_mask)
                cassava_mask_img = cassava_mask_img.resize(pred.shape[::-1], Image.NEAREST)
                cassava_mask = np.array(cassava_mask_img)
                
                # 计算预测中的腐烂区域
                pred_rotten_mask = (pred == 1).astype(np.uint8)
                
                # 只计算在木薯区域内的预测腐烂区域
                pred_rotten_in_cassava = pred_rotten_mask * cassava_mask
                
                # 计算面积
                cassava_pixels = np.count_nonzero(cassava_mask)
                pred_rotten_pixels = np.count_nonzero(pred_rotten_in_cassava)
                
                # 计算PPD值（预测腐烂面积/木薯面积）
                ppd = pred_rotten_pixels / cassava_pixels if cassava_pixels > 0 else 0.0
                
                # 添加调试信息
                if cassava_pixels == 0:
                    print(f"警告: 图像 {img_name} 未检测到木薯区域")
                
                # 存储结果用于汇总
                ppd_results.append({
                    'image_name': img_name + '.' + ext,
                    'pred_rotten_pixels': pred_rotten_pixels,
                    'cassava_pixels': cassava_pixels,
                    'ppd': ppd
                })
                
                # 创建对比图像
                comparison_img = create_comparison_image(original_img, pred, ppd)
                
                # 创建PPD分析可视化图像
                ppd_analysis_img = visualize_ppd_analysis(original_img, pred, cassava_mask, ppd)
                
                if save_val_results_to:
                    # 保存对比图像
                    comparison_img.save(os.path.join(save_val_results_to, img_name+'_comparison.png'))
                    # 保存PPD分析可视化图像
                    ppd_analysis_img.save(os.path.join(save_val_results_to, img_name+'_ppd_analysis.png'))
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue
        
        # 生成PPD汇总表
        if ppd_results and save_val_results_to:
            df = pd.DataFrame(ppd_results)
            df = df.sort_values('image_name')
            
            # 保存为CSV文件
            csv_path = os.path.join(save_val_results_to, 'ppd_summary.csv')
            df.to_csv(csv_path, index=False)
            
            # 计算统计信息
            avg_ppd = df['ppd'].mean()
            min_ppd = df['ppd'].min()
            max_ppd = df['ppd'].max()
            total_images = len(df)
            non_zero_ppd_count = len(df[df['ppd'] > 0])
            
            # 创建汇总统计信息
            summary_stats = {
                'total_images': [total_images],
                'images_with_rotten_areas': [non_zero_ppd_count],
                'average_ppd': [avg_ppd],
                'min_ppd': [min_ppd],
                'max_ppd': [max_ppd]
            }
            summary_df = pd.DataFrame(summary_stats)
            summary_csv_path = os.path.join(save_val_results_to, 'ppd_summary_stats.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            
            print(f"\nPPD汇总结果已保存到 {csv_path}")
            print(f"PPD统计信息已保存到 {summary_csv_path}")
            print(f"\n统计信息:")
            print(f"总图片数: {total_images}")
            print(f"包含腐烂区域的图片数: {non_zero_ppd_count}")
            print(f"平均PPD: {avg_ppd:.4f}")
            print(f"最小PPD: {min_ppd:.4f}")
            print(f"最大PPD: {max_ppd:.4f}")

if __name__ == '__main__':
    main()