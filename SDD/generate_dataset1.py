import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, binary_closing, binary_dilation
from PIL import Image
import random
from tqdm import tqdm

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
    # 确保kernel_size是奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # 创建坐标网格
    y, x = np.ogrid[-center:center+1, -center:center+1]
    
    # 旋转坐标系统，使phi方向对齐x轴
    x_rot = x * np.cos(phi) + y * np.sin(phi)
    y_rot = -x * np.sin(phi) + y * np.cos(phi)
    
    # 创建更窄的线条
    line_width = 1.0
    
    # 在旋转后的坐标系中，沿x轴方向创建线
    mask = (np.abs(x_rot) <= length/2) & (np.abs(y_rot) <= line_width)
    kernel[mask] = 1.0
    
    # 归一化
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()
    
    return kernel

def generate_star_background(image_size, num_stars, sigma, intensity_range=(30, 100)):
    """
    生成恒星背景 - 调小恒星大小和亮度
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
        
        # 计算PSF的有效范围 - 恒星小一点
        star_sigma = sigma * 0.3  # 使恒星更小
        radius = int(3 * star_sigma)
        x_min, x_max = max(0, x0-radius), min(image_size, x0+radius+1)
        y_min, y_max = max(0, y0-radius), min(image_size, y0+radius+1)
        
        # 应用PSF
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        X, Y = np.meshgrid(x_range, y_range)
        
        psf_values = point_spread_function(X, Y, x0, y0, S, star_sigma)
        image[y_min:y_max, x_min:x_max] += psf_values
    
    return image, gt_mask

def calculate_debris_position(start_position, frame, phi, velocity, image_size, kernel_size):
    """
    根据固定起始位置、帧号、运动方向和速度计算碎片在当前帧的位置
    确保碎片沿着条带延申方向移动
    
    start_position: 碎片的固定起始位置 (x, y)
    frame: 当前帧号
    phi: 运动方向（弧度）
    velocity: 碎片速度
    """
    # 计算当前帧的位移
    dx = velocity * frame * np.cos(phi)
    dy = velocity * frame * np.sin(phi)
    
    # 添加位移到起始位置
    x0 = int(start_position[0] + dx)
    y0 = int(start_position[1] + dy)
    
    # 确保在图像范围内
    margin = kernel_size // 2 + 5
    x0 = max(margin, min(image_size - margin - 1, x0))
    y0 = max(margin, min(image_size - margin - 1, y0))
    
    return x0, y0

def add_space_debris(image, gt_mask, num_debris, sigma, frame, sequence_snr, velocities, phis, image_size, start_positions, intensity_range=(50, 150)):
    """
    添加空间碎片 - 每个碎片有独立的运动方向
    num_debris: 碎片数量
    sigma: 镜头弥撒半径
    frame: 当前帧号
    sequence_snr: 序列的信噪比级别
    velocities: 每个碎片的速度
    phis: 每个碎片的运动方向（弧度）
    start_positions: 每个碎片的固定起始位置
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
        length = velocity
        
        # 调整kernel大小
        kernel_size = min(image_size//4, max(21, int(length * 2)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建运动模糊核 - phi方向就是条带延申方向
        kernel = motion_blur_kernel(length, phi, kernel_size)
        
        # 计算碎片在当前帧的位置 - 沿着phi方向移动
        x0, y0 = calculate_debris_position(start_positions[i], frame, phi, velocity, image_size, kernel_size)
        
        # 碎片亮度
        S = random.uniform(intensity_range[0], intensity_range[1])
        
        # 计算PSF的有效范围
        debris_sigma = sigma * 0.3  # 使碎片略小于默认大小
        radius = int(2 * debris_sigma)
        x_min, x_max = max(0, x0-radius), min(image_size, x0+radius+1)
        y_min, y_max = max(0, y0-radius), min(image_size, y0+radius+1)
        
        # 创建点光源
        point_source = np.zeros((image_size, image_size))
        if x_max > x_min and y_max > y_min:  # 确保范围有效
            x_range = np.arange(x_min, x_max)
            y_range = np.arange(y_min, y_max)
            X, Y = np.meshgrid(x_range, y_range)
            
            psf_values = point_spread_function(X, Y, x0, y0, S, debris_sigma)
            point_source[y_min:y_max, x_min:x_max] = psf_values
            
            # 应用运动模糊 - 方向与phi一致
            blurred_source = convolve(point_source, kernel, mode='constant')
            
            # 添加到干净图像
            clean_image += blurred_source
            
            # 创建临时掩码
            temp_mask = np.zeros((image_size, image_size), dtype=np.uint8)
            
            # 使用阈值生成初始掩码
            threshold = 0.2 * blurred_source.max()
            initial_mask = blurred_source > threshold
            
            # 应用形态学闭操作连接分段部分
            # 创建结构元素，大小根据碎片运动长度调整
            struct_size = max(3, min(15, int(velocity / 3)))
            # 确保struct_size是奇数
            if struct_size % 2 == 0:
                struct_size += 1
                
            # 创建与运动方向一致的结构元素
            struct = np.zeros((struct_size, struct_size), dtype=bool)
            center = struct_size // 2
            for x in range(struct_size):
                for y in range(struct_size):
                    dx = x - center
                    dy = y - center
                    # 计算点到直线的距离
                    dist = abs(dx * np.sin(phi) - dy * np.cos(phi))
                    # 在直线方向上延伸
                    if dist <= 1:
                        struct[y, x] = True
            
            # 应用闭操作
            closed_mask = binary_closing(initial_mask, structure=struct, iterations=2)
            
            # 再次膨胀以确保连续性
            final_mask = binary_dilation(closed_mask, structure=struct, iterations=1)
            
            # 转换为uint8格式
            temp_mask[final_mask] = 255
            temp_masks.append(temp_mask)
    
    # 合并所有碎片掩码
    for mask in temp_masks:
        gt_mask = np.maximum(gt_mask, mask)
    
    return clean_image, gt_mask

def add_sequence_noise(image, snr):
    """
    为整个图像添加统一的高斯噪声
    snr: 信噪比
    """
    # 计算噪声水平
    signal_power = np.mean(image**2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)
    
    # 添加随机扰动 (最多5%)
    disturbance_percentage = random.uniform(0, 0.05)
    disturbance = disturbance_percentage * image.max() * np.random.randn(*image.shape)
    
    # 返回带噪声的图像
    return image + noise + disturbance

def save_image(image, path):
    """保存图像为PNG格式"""
    # 归一化到0-255范围
    normalized = np.clip(image, 0, None)
    normalized = (normalized / normalized.max() * 255).astype(np.uint8)
    
    # 保存图像
    Image.fromarray(normalized).save(path)

def generate_dataset():
    """生成整个数据集"""
    # 参数列表
    snr_levels = [1.25, 1.50, 1.75, 2.00, 2.5, 3.00, 5.00, 10.00]  # 八个信噪比级别
    velocity_levels = [5, 10, 15, 20, 25]  # 五个线速度级别
    sigma_levels = [1.25, 1.50, 1.75, 2.00, 2.25]  # 五个弥撒半径级别
    
    image_size = 512
    frames_per_sequence = 5
    
    # 创建目录结构
    base_dir = "v3"
    
    # 计算总序列数
    total_sequences = len(sigma_levels) * 30  # 每个sigma级别30个序列，共150个
    
    # 确定训练和测试集划分
    test_count = int(total_sequences * 0.3)
    sequence_indices = list(range(total_sequences))
    random.shuffle(sequence_indices)
    test_indices = set(sequence_indices[:test_count])
    
    # 生成所有序列
    sequence_idx = 0
    
    # 为每个sigma值生成30个序列
    for sigma in tqdm(sigma_levels, desc="Sigma Levels"):
        for _ in tqdm(range(30), desc=f"Sequences for sigma={sigma}", leave=False):
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
            num_stars = random.randint(100, 300)
            star_background, _ = generate_star_background(image_size, num_stars, sigma)
            
            # 为序列随机选择一个信噪比
            sequence_snr = random.choice(snr_levels)
            
            # 确定该序列中碎片的数量
            num_debris = random.randint(5, 10)
            
            # 为每个碎片分配速度
            debris_velocities = [random.choice(velocity_levels) for _ in range(num_debris)]
            
            # 为每个碎片分配独立的运动方向
            debris_phis = [random.uniform(0, 2 * np.pi) for _ in range(num_debris)]
            
            # 预先生成每个碎片的起始位置，确保在整个序列中位置连续
            # 在图像中心区域随机分布起始位置
            center_x, center_y = image_size // 2, image_size // 2
            start_positions = []
            
            for i in range(num_debris):
                # 计算最大移动距离
                max_move = debris_velocities[i] * frames_per_sequence
                phi = debris_phis[i]  # 使用碎片自己的运动方向
                
                # 随机生成起始位置，确保整个运动轨迹在图像内
                margin = 50  # 边界安全距离
                valid_position = False
                attempts = 0
                
                # 尝试找到有效的起始位置
                while not valid_position and attempts < 20:
                    # 在中心区域随机选择点
                    r = random.randint(50, 150)  # 距中心的随机距离
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
                    
                    attempts += 1
                
                # 如果找不到有效位置，使用中心区域的安全位置
                if not valid_position:
                    # 反向计算起始位置，使运动方向朝向图像中心
                    start_x = center_x - (max_move / 2) * np.cos(phi)
                    start_y = center_y - (max_move / 2) * np.sin(phi)
                
                start_positions.append((start_x, start_y))
            
            # 生成序列的每一帧
            for frame in range(frames_per_sequence):
                # 复制恒星背景
                frame_image = star_background.copy()
                frame_gt = np.zeros((image_size, image_size), dtype=np.uint8)
                
                # 添加空间碎片，使用固定的起始位置和每个碎片独立的运动方向
                clean_image, frame_gt = add_space_debris(
                    frame_image, frame_gt, num_debris, sigma, 
                    frame, sequence_snr, debris_velocities, debris_phis, image_size,
                    start_positions
                )
                
                # 添加序列一致的噪声
                noisy_image = add_sequence_noise(clean_image, sequence_snr)
                
                # 保存图像和地面真值
                image_path = os.path.join(seq_folder, f"{frame:03d}.png")
                gt_path = os.path.join(seq_folder_gt, f"{frame:03d}.png")
                
                save_image(noisy_image, image_path)
                Image.fromarray(frame_gt).save(gt_path)
            
            sequence_idx += 1

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    print("开始生成空间碎片仿真数据集...")
    generate_dataset()
    print("数据集生成完成！")
