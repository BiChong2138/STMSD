import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, binary_closing, binary_dilation, gaussian_filter, binary_erosion, binary_opening
from PIL import Image
import random
from tqdm import tqdm
import cv2

def create_directory(path):
    """创建目录，如果目录不存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def point_spread_function(x, y, x0, y0, S, sigma):
    """
    点扩散函数 (PSF) 模型
    (x0, y0): 光源质心
    S: 从光能量到灰度级的转换系数
    sigma: 观测系统镜头的弥撒半径
    """
    return S / (2 * np.pi * sigma**2) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def motion_blur_kernel(length, phi, kernel_size):
    """
    运动模糊核 - 确保生成连续的条状而不是离散点
    length: 运动长度
    phi: 运动方向（弧度）
    kernel_size: 核大小
    """
    # 确保kernel_size是奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # 为低速情况设定最小长度，确保方向性明显
    effective_length = max(3.0, length)

    # 创建坐标网格
    y, x = np.ogrid[-center:center+1, -center:center+1]
    
    # 旋转坐标系统，使phi方向对齐x轴
    x_rot = x * np.cos(phi) + y * np.sin(phi)
    y_rot = -x * np.sin(phi) + y * np.cos(phi)
    
    # 创建线条粗细 - 确保连续性
    line_width = max(1.0, effective_length/5)  # 线宽与长度相关，确保连续性
    
    # 在旋转后的坐标系中，沿x轴方向创建线
    mask = (np.abs(x_rot) <= effective_length/2) & (np.abs(y_rot) <= line_width/2)
    kernel[mask] = 1.0
    
    # 归一化
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()
    
    return kernel

def directional_structure_element(phi, size, length, is_short=False, is_long=False):
    """
    创建严格方向性的结构元素，保持条状方向一致性
    phi: 方向角（弧度）
    size: 结构元素大小
    length: 碎片长度
    is_short: 是否为短碎片
    is_long: 是否为长碎片（进一步延长结构元素）
    """
    # 确保size是奇数
    if size % 2 == 0:
        size += 1
        
    struct = np.zeros((size, size), dtype=bool)
    center = size // 2
    
    # 对短碎片使用更严格的方向控制，长碎片延长结构元素
    if is_short:
        # 短碎片使用严格的线状结构元素，完全对齐phi方向
        width_multiplier = 0.5  # 更窄的结构
    elif is_long:
        # 长碎片使用更长的结构元素
        width_multiplier = 0.9  # 稍宽但更长
    else:
        # 中等碎片可以稍宽
        width_multiplier = 0.8
        
    # 计算线宽，与长度成反比
    line_width = max(0.6, size/6) * width_multiplier / (1 + length * 0.05)
    
    # 长碎片结构元素放大比例
    length_scale = 1.0
    if is_long:
        length_scale = 1.1  # 增加长碎片结构元素的长度
    
    # 创建严格的方向性结构元素
    for y in range(size):
        for x in range(size):
            # 相对中心的坐标
            dx = x - center
            dy = y - center
            
            # 计算点到直线的距离，使用参数方程
            dist = abs(dx * np.sin(phi) - dy * np.cos(phi))
            
            # 使用更严格的距离阈值以保持方向性
            if dist <= line_width:
                # 检查点到原点的距离是否在结构元素长度范围内
                dist_to_center = np.sqrt(dx**2 + dy**2)
                max_length = center * 0.95 * length_scale  # 长碎片使用更长结构元素
                
                # 计算点在方向线上的投影
                projection = dx * np.cos(phi) + dy * np.sin(phi)
                
                # 只有当点的投影在方向线上且距离足够小时才设为True
                if is_long:
                    # 长碎片的投影范围放宽
                    projection_limit = max_length * 1.1
                else:
                    projection_limit = max_length
                    
                if dist_to_center <= max_length and abs(projection) <= projection_limit:
                    struct[y, x] = True
    
    # 确保结构元素至少有一个点为True
    if not np.any(struct):
        struct[center, center] = True
        
        # 添加方向点
        dx = int(round(np.cos(phi)))
        dy = int(round(np.sin(phi)))
        
        if 0 <= center+dy < size and 0 <= center+dx < size:
            struct[center+dy, center+dx] = True
        
        # 对于长碎片，尝试再添加一个更远的点以延长结构
        if is_long:
            dx2 = int(round(2 * np.cos(phi)))
            dy2 = int(round(2 * np.sin(phi)))
            if 0 <= center+dy2 < size and 0 <= center+dx2 < size:
                struct[center+dy2, center+dx2] = True
    
    return struct

def generate_star_background(image_size, num_stars, sigma, intensity_range=(30, 100)):
    """
    生成恒星背景 - 调整恒星大小和亮度
    num_stars: 恒星数量
    sigma: 镜头弥撒半径
    intensity_range: 亮度范围
    """
    image = np.zeros((image_size, image_size))
    gt_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    for _ in range(num_stars):
        x0 = random.randint(0, image_size-1)
        y0 = random.randint(0, image_size-1)
        S = random.uniform(intensity_range[0], intensity_range[1])
        
        # 计算PSF的有效范围 - 调大恒星尺寸
        star_sigma = sigma * 0.4  # 从0.3增加到0.4，使恒星略大
        radius = max(1, int(3 * star_sigma))  # 确保半径至少为1
        # 确保坐标不超出边界
        x_min, x_max = max(0, x0-radius), min(image_size, x0+radius+1)
        y_min, y_max = max(0, y0-radius), min(image_size, y0+radius+1)
        
        # 确保区域大小有效
        if x_max <= x_min or y_max <= y_min:
            continue
            
        # 应用PSF
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        X, Y = np.meshgrid(x_range, y_range)
        
        psf_values = point_spread_function(X, Y, x0, y0, S, star_sigma)
        image[y_min:y_max, x_min:x_max] += psf_values
    
    # 确保边缘区域没有异常亮度
    # 创建边缘掩码
    edge_mask = np.ones((image_size, image_size), dtype=bool)
    edge_width = 2
    edge_mask[edge_width:-edge_width, edge_width:-edge_width] = False
    
    # 如果边缘区域的值异常高，将其设置为内部区域的平均值
    if np.any(edge_mask):
        inner_mean = np.mean(image[~edge_mask])
        image[edge_mask] = np.minimum(image[edge_mask], inner_mean)
    
    return image, gt_mask

def calculate_debris_position(start_position, frame, phi, velocity, image_size, kernel_size):
    """
    根据固定起始位置、帧号、运动方向和速度计算碎片在当前帧的位置
    使用亚像素精度来计算位置
    
    start_position: 碎片的固定起始位置 (x, y)
    frame: 当前帧号
    phi: 运动方向（弧度）
    velocity: 碎片速度
    """
    # 计算当前帧的位移，使用浮点数保持亚像素精度
    dx = velocity * frame * np.cos(phi)
    dy = velocity * frame * np.sin(phi)
    
    # 添加位移到起始位置，保持浮点数精度
    x0 = start_position[0] + dx
    y0 = start_position[1] + dy
    
    # 确保在图像范围内
    margin = kernel_size // 2 + 5
    x0 = max(margin, min(image_size - margin - 1, x0))
    y0 = max(margin, min(image_size - margin - 1, y0))
    
    return x0, y0

def add_space_debris(image, gt_mask, num_debris, sigma, frame, debris_snrs, velocities, phis, image_size, start_positions, intensity_range=(50, 150)):
    """
    添加空间碎片 - 使用亚像素精度和更平滑的插值方法
    精确掩码生成，确保掩码与实际碎片位置精确匹配，保持方向一致性
    
    debris_snrs: 每个碎片的SNR值列表
    """
    # 创建清晰图像（不含噪声）
    clean_image = image.copy()
    
    # 创建临时掩码数组，用于存储每个碎片的单独掩码
    temp_masks = []
    
    for i in range(num_debris):
        # 获取当前碎片的速度和方向
        velocity = velocities[i]
        phi = phis[i]
        
        # 计算运动长度
        length = max(3, velocity)  # 确保最小长度
        
        # 调整kernel大小
        kernel_size = min(image_size//4, max(15, int(length * 2.0)))  # 略微减小系数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建运动模糊核
        kernel = motion_blur_kernel(length, phi, kernel_size)
        
        # 计算碎片在当前帧的亚像素精度位置
        x0, y0 = calculate_debris_position(start_positions[i], frame, phi, velocity, image_size, kernel_size)
        
        # 碎片亮度 - 基础亮度根据速度调整（保留这部分以获得初始亮度）
        base_intensity = random.uniform(intensity_range[0], intensity_range[1])
        if velocity < 1.0:
            # 低速碎片略微降低亮度（聚集在小区域）
            base_intensity *= 0.95
        elif velocity > 5.0:
            # 高速碎片略微提高亮度（覆盖更大区域）
            base_intensity *= 1.1
            
        # 先使用基础亮度生成初始碎片图像
        S = base_intensity
        
        # 计算PSF的有效范围 - 保持一致的大小
        debris_sigma = sigma * 0.35  # 调整大小适中
        radius = max(2, int(2.5 * debris_sigma))  # 确保点源足够但不过大
        
        # 使用亚像素精度创建网格
        # 计算整数坐标范围
        x_min_int, x_max_int = max(0, int(x0)-radius), min(image_size, int(x0)+radius+1)
        y_min_int, y_max_int = max(0, int(y0)-radius), min(image_size, int(y0)+radius+1)
        
        # 创建点光源
        point_source = np.zeros((image_size, image_size))
        if x_max_int > x_min_int and y_max_int > y_min_int:  # 确保范围有效
            x_range = np.arange(x_min_int, x_max_int)
            y_range = np.arange(y_min_int, y_max_int)
            X, Y = np.meshgrid(x_range, y_range)
            
            # 使用亚像素精度计算PSF值
            psf_values = point_spread_function(X, Y, x0, y0, S, debris_sigma)
            point_source[y_min_int:y_max_int, x_min_int:x_max_int] = psf_values
            
            # 应用运动模糊 - 修复边界处理问题
            kernel_float = kernel.astype(np.float32)
            
            # 添加边界填充，避免边缘伪影
            pad_size = kernel_size // 2
            point_source_padded = cv2.copyMakeBorder(
                point_source, 
                pad_size, pad_size, pad_size, pad_size, 
                cv2.BORDER_CONSTANT, 
                value=0
            )
            
            # 使用填充后的图像进行卷积
            blurred_padded = cv2.filter2D(
                point_source_padded.astype(np.float32), 
                -1, kernel_float, 
                borderType=cv2.BORDER_CONSTANT
            )
            
            # 去除填充，恢复原始尺寸
            blurred_source = blurred_padded[pad_size:pad_size+image_size, pad_size:pad_size+image_size]
            
            # 可选：使用高斯滤波进一步平滑
            blurred_source = gaussian_filter(blurred_source, sigma=0.5)
            
            # 保存未修改的原始模糊源以用于掩码生成
            original_blurred = blurred_source.copy()
            
            # 创建临时掩码
            temp_mask = np.zeros((image_size, image_size), dtype=np.uint8)
            
            # 判断碎片类型
            is_short_debris = length < 5
            is_long_debris = length > 6  # 识别较长碎片，需要特殊处理
            
            # 先生成临时掩码用于区分前景和背景
            # 使用自适应阈值以更精确地捕捉碎片形状
            intensity_factor = min(1.0, max(0.7, 0.7 + velocity * 0.06))
            if is_long_debris:
                intensity_factor *= 0.95
            intensity_max = original_blurred.max() * intensity_factor
            
            # 自适应阈值 - 根据碎片长度和速度动态调整
            if is_short_debris:
                angle_mod = abs(np.sin(2*phi))
                angle_factor = 1.0 + angle_mod * 0.1
                base_threshold = (0.21 if velocity < 3 else 0.23) * angle_factor
            elif is_long_debris:
                base_threshold = 0.15 if velocity < 3 else 0.18
            else:
                base_threshold = 0.17 if velocity < 3 else 0.20
                
            length_factor = min(1.1, 1.0 + (length - 5) * 0.02)
            if is_long_debris:
                length_factor *= 0.95
            threshold = base_threshold * length_factor * intensity_max
            
            # 生成前景掩码
            temp_mask_for_snr = original_blurred > threshold
            
            # 计算背景区域统计量（非碎片区域）
            background_mask = ~temp_mask_for_snr
            if np.sum(background_mask) > 0:  # 确保有背景像素
                # 使用原始图像（包括恒星背景）计算背景统计量
                background_mean = np.mean(clean_image[background_mask])
                background_std = max(0.1, np.std(clean_image[background_mask]))
            else:
                # 防止出现无背景像素的极端情况
                background_mean = np.mean(clean_image)
                background_std = max(0.1, np.std(clean_image))
            
            # 计算当前前景区域均值
            if np.sum(temp_mask_for_snr) > 0:  # 确保有前景像素
                foreground_mean = np.mean(original_blurred[temp_mask_for_snr])
            else:
                foreground_mean = np.max(original_blurred) if np.max(original_blurred) > 0 else 1.0
            
            # 根据目标SNR计算需要的前景均值
            target_snr = debris_snrs[i]
            # SNR = (信号 - 背景) / 背景噪声
            required_foreground_mean = background_mean + target_snr * background_std
            
            # 计算亮度调整系数
            if foreground_mean > 0:
                snr_intensity_factor = required_foreground_mean / foreground_mean
            else:
                snr_intensity_factor = 1.0
            
            # 限制过大和过小的调整因子，防止极端值
            snr_intensity_factor = min(max(snr_intensity_factor, 0.1), 5.0)
            
            # 应用亮度调整
            blurred_source *= snr_intensity_factor
            
            # 添加到干净图像
            clean_image += blurred_source
            
            # 根据新的亮度调整后的模糊源重新生成掩码
            # 保存调整后的模糊源用于掩码生成
            adjusted_blurred = blurred_source.copy()
            
            # 生成掩码 - 使用精确阈值
            # 由于亮度发生变化，需要重新计算阈值
            adjusted_intensity_max = adjusted_blurred.max() * intensity_factor
            adjusted_threshold = base_threshold * length_factor * adjusted_intensity_max
            
            initial_mask = adjusted_blurred > adjusted_threshold
            
            # 创建与运动方向严格一致的精确结构元素
            if is_short_debris:
                # 短碎片使用较小但严格方向性的结构元素
                struct_size = max(3, min(5, int(length/3)))
            elif is_long_debris:
                # 长碎片使用稍大的结构元素以延长掩码
                struct_size = max(5, min(7, int(length/3)))
            else:
                # 中等碎片使用标准大小
                struct_size = max(3, min(5, int(length/4)))
                
            # 使用专用函数创建严格方向性的结构元素，长碎片使用更长的结构元素
            struct = directional_structure_element(phi, struct_size, length, 
                                                  is_short=is_short_debris,
                                                  is_long=is_long_debris)
            
            # 应用闭操作，保持方向一致性
            iterations = 1
            
            # 长碎片可以用多一次的迭代来延长掩码
            if is_long_debris and velocity > 5:
                iterations = 2
                
            closed_mask = binary_closing(initial_mask, structure=struct, iterations=iterations)
            
            # 对长碎片进行额外处理以延长掩码 - 移除膨胀操作
            if is_long_debris:
                final_mask = closed_mask
            elif not is_short_debris and length > 5:
                # 中长碎片应用轻微腐蚀
                erosion_struct = np.ones((2, 2), dtype=bool)
                final_mask = binary_erosion(closed_mask, structure=erosion_struct, iterations=1)
            else:
                final_mask = closed_mask
            
            # 可选的最终处理：保持条状连续性
            if is_short_debris:
                # 使用形态学操作保持方向性
                # 创建严格的线状结构元素
                line_struct = np.zeros((3, 3), dtype=bool)
                center = 1
                
                # 创建沿phi方向的线状结构元素
                dx = np.cos(phi)
                dy = np.sin(phi)
                
                # 设置中心点
                line_struct[center, center] = True
                
                # 设置方向点
                next_x = center + int(round(dx))
                next_y = center + int(round(dy))
                if 0 <= next_y < 3 and 0 <= next_x < 3:
                    line_struct[next_y, next_x] = True
                
                prev_x = center - int(round(dx))
                prev_y = center - int(round(dy))
                if 0 <= prev_y < 3 and 0 <= prev_x < 3:
                    line_struct[prev_y, prev_x] = True
                
                # 应用方向性开操作，移除垂直于运动方向的扩展
                final_mask = binary_opening(final_mask, structure=line_struct, iterations=1)
            
            # 对中等和长碎片应用平滑处理，短碎片不进行平滑以保留细节
            if not is_short_debris:
                # 使用适度的高斯平滑，然后重新二值化
                smoothed_mask = gaussian_filter(final_mask.astype(float), sigma=0.5)
                final_mask = smoothed_mask > 0.5
            
            # 转换为uint8格式
            temp_mask[final_mask] = 255
            temp_masks.append(temp_mask)
    
    # 合并所有碎片掩码
    for mask in temp_masks:
        gt_mask = np.maximum(gt_mask, mask)
    
    # 返回图像和掩码
    return clean_image, gt_mask

def add_uniform_noise(image, base_noise_level=0.1, variation_level=0.05):
    """
    为整个图像添加两层噪声：
    1. 基础噪声层：模拟成像系统固有噪声、杂散光、Johnson-Nyquist噪声等
    2. 变化扰动层：模拟真实世界观测条件的小变化（不超过5%）
    
    image: 输入图像
    base_noise_level: 基础噪声水平
    variation_level: 额外变化扰动的最大幅度
    """
    # 计算图像信号强度
    image_mean = max(0.1, np.mean(image))
    
    # 第一层：基础系统噪声 - 使用固定水平
    base_noise_std = image_mean * base_noise_level
    base_noise = np.random.normal(0, base_noise_std, image.shape)
    
    # 应用基础噪声
    noisy_image = image + base_noise
    
    # 第二层：随机变化扰动 - 不超过5%的随机变化
    # 生成随机变化幅度 (0-5%)
    variation_factor = random.uniform(0, variation_level)
    variation_std = image_mean * variation_factor
    
    # 生成变化扰动
    variation_noise = np.random.normal(0, variation_std, image.shape)
    
    # 应用变化扰动
    final_image = noisy_image + variation_noise
    
    return final_image

def check_trajectory_overlap(positions_list, velocities, phis, frames, image_size, min_distance=10):
    """
    检查碎片轨迹是否重叠 - 使用亚像素精度
    
    positions_list: 所有碎片的起始位置列表
    velocities: 所有碎片的速度列表
    phis: A所有碎片的方向列表
    frames: 序列的帧数
    min_distance: 允许的最小距离
    
    返回: 如果轨迹重叠返回True，否则返回False
    """
    if len(positions_list) < 2:  # 少于2个碎片，不存在重叠
        return False
    
    # 预先计算所有帧中所有碎片的位置 - 使用浮点数保持亚像素精度
    all_positions = []
    
    for i, start_pos in enumerate(positions_list):
        debris_positions = []
        velocity = velocities[i]
        phi = phis[i]
        
        for frame in range(frames):
            # 计算当前帧中的位置
            dx = velocity * frame * np.cos(phi)
            dy = velocity * frame * np.sin(phi)
            
            pos_x = start_pos[0] + dx
            pos_y = start_pos[1] + dy
            
            # 确保在图像范围内
            pos_x = max(0, min(image_size - 1, pos_x))
            pos_y = max(0, min(image_size - 1, pos_y))
            
            debris_positions.append((pos_x, pos_y))
        
        all_positions.append(debris_positions)
    
    # 检查每对碎片在每帧中的距离
    for i in range(len(all_positions)):
        for j in range(i+1, len(all_positions)):
            for frame in range(frames):
                pos1 = all_positions[i][frame]
                pos2 = all_positions[j][frame]
                
                # 计算欧几里得距离
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < min_distance:
                    return True  # 轨迹重叠
    
    return False  # 没有重叠

def save_image(image, path):
    """保存图像为PNG格式 - 使用标准归一化"""
    # 使用简单的标准归一化，保持所有像素的相对关系
    # 这样可以保持SNR不变，因为前景和背景会被等比例缩放
    normalized = np.clip(image, 0, None)
    
    # 避免除零错误
    if normalized.max() > 0:
        normalized = (normalized / normalized.max() * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    
    # 保存图像
    Image.fromarray(normalized).save(path)

def generate_dataset():
    """生成整个数据集"""
    # 参数列表
    snr_levels = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]  # 更新的信噪比级别
    velocity_levels = [0.5, 1, 3, 5, 7]  # 五个线速度级别，调整以避免极低速度问题
    sigma_levels = [1.25, 1.50, 1.75, 2.00, 2.25]  # 五个弥撒半径级别
    
    # 噪声参数 - 为不同的序列随机选择基础噪声水平
    base_noise_levels = [0.1, 0.12, 0.15, 0.18]  # 基础噪声水平选项
    image_size = 256  # 256×256分辨率
    frames_per_sequence = 5
    
    # 创建目录结构
    base_dir = "v12"
    
    # 计算总序列数
    sequences_per_sigma = 120  # 每个sigma级别120个序列
    total_sequences = len(sigma_levels) * sequences_per_sigma  # 共600个序列
    
    # 确定训练和测试集划分（70%训练，30%测试）
    test_count = int(total_sequences * 0.3)  # 30%用于测试
    sequence_indices = list(range(total_sequences))
    random.shuffle(sequence_indices)
    test_indices = set(sequence_indices[:test_count])
    
    # 生成所有序列
    sequence_idx = 0
    
    # 为每个sigma值生成120个序列
    for sigma in tqdm(sigma_levels, desc="Sigma Levels"):
        for seq_num in tqdm(range(sequences_per_sigma), desc=f"Sequences for sigma={sigma}", leave=False):
            # 确定是训练集还是测试集
            is_test = sequence_idx in test_indices
            main_folder = "Test" if is_test else "Train"
            main_folder_gt = "Test_GT" if is_test else "Train_GT"
            
            # 序列编号 (000, 001, ...)
            sequence_name = f"{sequence_idx:03d}"
            
            # 创建序列文件夹
            seq_folder = os.path.join(base_dir, main_folder, sequence_name)
            seq_folder_gt = os.path.join(base_dir, main_folder_gt, sequence_name)
            create_directory(seq_folder)
            create_directory(seq_folder_gt)
            
            # 为序列生成恒星背景（在一个序列中保持不变）
            num_stars = random.randint(60, 150)  # 适应256x256分辨率的恒星数量
            star_background, _ = generate_star_background(image_size, num_stars, sigma)
            
            # 确保背景完全为零
            bg_threshold = 1e-6
            star_background[star_background < bg_threshold] = 0.0
            
            # 确定该序列中碎片的数量
            num_debris = random.randint(3, 7)  # 适应分辨率的碎片数量
            
            # 为每个碎片分配信噪比 - 确保SNR等级均匀分布
            debris_snrs = []
            for i in range(num_debris):
                # 使用(sequence_idx + i)确保每个碎片的SNR分布均匀
                snr_index = (sequence_idx + i) % len(snr_levels)
                snr = snr_levels[snr_index]
                
                # 为保证物理合理性，增加随机扰动（±5%）
                snr_variation = random.uniform(0.95, 1.05)
                final_snr = snr * snr_variation
                
                debris_snrs.append(final_snr)
            
            # 为每个碎片分配速度 - 确保速度等级均匀分布
            debris_velocities = []
            for i in range(num_debris):
                # 使用(sequence_idx + i)确保每个碎片的速度分布均匀
                v_index = (sequence_idx + i) % len(velocity_levels)
                velocity = velocity_levels[v_index]
                
                # 为保证物理合理性，增加随机扰动（±10%）
                velocity_variation = random.uniform(0.9, 1.1)
                final_velocity = velocity * velocity_variation
                
                debris_velocities.append(final_velocity)
            
            # 为每个碎片分配独立的运动方向
            debris_phis = [random.uniform(0, 2 * np.pi) for _ in range(num_debris)]
            
            # 预先生成每个碎片的起始位置，确保在整个序列中位置连续且不重叠
            # 在图像中心区域随机分布起始位置
            center_x, center_y = image_size // 2, image_size // 2
            
            # 循环尝试生成不重叠的轨迹组合
            max_attempts = 30  # 最大尝试次数
            attempt_count = 0
            valid_configuration = False
            
            # 在反复尝试失败后可能需要减少碎片数量
            adjusted_num_debris = num_debris
            
            while not valid_configuration and attempt_count < max_attempts:
                start_positions = []
                
                # 生成所有碎片的起始位置
                for i in range(adjusted_num_debris):
                    # 计算最大移动距离
                    max_move = debris_velocities[i] * frames_per_sequence
                    phi = debris_phis[i]  # 使用碎片自己的运动方向
                    
                    # 随机生成起始位置，确保整个运动轨迹在图像内
                    margin = 25  # 边界安全距离
                    valid_position = False
                    pos_attempts = 0
                    
                    # 尝试找到有效的起始位置
                    while not valid_position and pos_attempts < 20:
                        # 在中心区域随机选择点
                        r = random.randint(25, 75)  # 距中心的随机距离
                        theta = random.uniform(0, 2 * np.pi)  # 随机角度
                        
                        start_x = center_x + r * np.cos(theta)
                        start_y = center_y + r * np.sin(theta)
                        
                        # 计算最终位置
                        end_x = start_x + max_move * np.cos(phi)
                        end_y = start_y + max_move * np.sin(phi)
                        
                        # 检查轨迹是否在图像内
                        if (margin <= start_x <= image_size - margin and
                            margin <= start_y <= image_size - margin and
                            margin <= end_x <= image_size - margin and
                            margin <= end_y <= image_size - margin):
                            valid_position = True
                        
                        pos_attempts += 1
                    
                    # 如果找不到有效位置，使用中心区域的安全位置
                    if not valid_position:
                        # 反向计算起始位置，使运动方向朝向图像中心
                        start_x = center_x - (max_move / 2) * np.cos(phi)
                        start_y = center_y - (max_move / 2) * np.sin(phi)
                    
                    start_positions.append((start_x, start_y))
                
                # 检查轨迹是否重叠
                if not check_trajectory_overlap(start_positions, debris_velocities[:adjusted_num_debris], 
                                              debris_phis[:adjusted_num_debris], frames_per_sequence, 
                                              image_size, min_distance=12):
                    valid_configuration = True
                else:
                    attempt_count += 1
                    # 如果多次尝试失败，减少碎片数量
                    if attempt_count == max_attempts // 2 and adjusted_num_debris > 3:
                        adjusted_num_debris -= 1
                        attempt_count = 0  # 重置尝试次数
                        print(f"Reducing debris count to {adjusted_num_debris} for sequence {sequence_name}")
            
            # 如果仍然无法找到不重叠的配置，使用最后生成的位置
            num_debris = adjusted_num_debris  # 更新实际使用的碎片数量
            # 截取相应数量的参数列表
            debris_velocities = debris_velocities[:num_debris]
            debris_phis = debris_phis[:num_debris]
            debris_snrs = debris_snrs[:num_debris]
            
            # 为该序列随机选择基础噪声水平
            base_noise_level = random.choice(base_noise_levels)
            
            # 生成序列的每一帧
            for frame in range(frames_per_sequence):
                # 复制恒星背景
                frame_image = star_background.copy()
                frame_gt = np.zeros((image_size, image_size), dtype=np.uint8)
                
                # 添加空间碎片，使用固定的起始位置和每个碎片独立的运动方向和SNR
                clean_image, frame_gt = add_space_debris(
                    frame_image, frame_gt, num_debris, sigma, 
                    frame, debris_snrs, debris_velocities, debris_phis, image_size,
                    start_positions
                )
                
                # 添加两层噪声：基础系统噪声和随机变化扰动
                noisy_image = add_uniform_noise(clean_image, 
                                              base_noise_level=base_noise_level,
                                              variation_level=0.05)
                
                # 保存图像和地面真值
                image_path = os.path.join(seq_folder, f"{frame:03d}.png")
                gt_path = os.path.join(seq_folder_gt, f"{frame:03d}.png")
                
                save_image(noisy_image, image_path)
                Image.fromarray(frame_gt).save(gt_path)
            
            sequence_idx += 1

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(43)
    np.random.seed(43)
    
    print("开始生成空间碎片仿真数据集...")
    generate_dataset()
    print("数据集生成完成！")
