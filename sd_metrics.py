import numpy as np
from skimage import measure
from scipy.spatial.distance import cdist
import torch

def extract_instances(mask):
    """
    从分割掩码中提取连通区域作为目标实例，并计算质心
    
    参数:
        mask: 二值分割掩码 (numpy array)
    
    返回:
        instances: 列表，包含每个实例的标签和属性
        centroids: numpy array，每个实例的质心坐标 [n_instances, 2]
    """
    # 标记连通区域
    labeled_mask = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    
    instances = []
    centroids = []
    
    # 提取每个区域的属性
    for region in regions:
        instances.append({
            'label': region.label,
            'area': region.area,
            'bbox': region.bbox,
            'centroid': region.centroid
        })
        centroids.append(region.centroid)
    
    return instances, np.array(centroids) if centroids else np.empty((0, 2))

def match_instances(pred_centroids, gt_centroids, distance_threshold):
    """
    基于质心距离匹配预测实例和真实实例
    
    参数:
        pred_centroids: 预测实例的质心坐标 [n_pred, 2]
        gt_centroids: 真实实例的质心坐标 [n_gt, 2]
        distance_threshold: 判定为成功匹配的最大距离阈值
    
    返回:
        matches: 列表，包含匹配的(预测索引, 真实索引)对
        unmatched_preds: 列表，未匹配的预测索引
        unmatched_gts: 列表，未匹配的真实索引
    """
    matches = []
    unmatched_preds = list(range(len(pred_centroids)))
    unmatched_gts = list(range(len(gt_centroids)))
    
    # 如果预测或真实实例为空，直接返回
    if len(pred_centroids) == 0 or len(gt_centroids) == 0:
        return matches, unmatched_preds, unmatched_gts
    
    # 计算所有预测与真实质心之间的距离矩阵
    distance_matrix = cdist(pred_centroids, gt_centroids)
    
    # 对每个预测实例找到最近的真实实例
    for pred_idx in range(len(pred_centroids)):
        if len(unmatched_gts) == 0:
            break
            
        # 找到当前预测实例与所有未匹配真实实例的距离
        distances = distance_matrix[pred_idx, unmatched_gts]
        closest_gt_idx = unmatched_gts[np.argmin(distances)]
        min_distance = distance_matrix[pred_idx, closest_gt_idx]
        
        # 如果距离小于阈值，认为是成功匹配
        if min_distance < distance_threshold:
            matches.append((pred_idx, closest_gt_idx))
            unmatched_preds.remove(pred_idx)
            unmatched_gts.remove(closest_gt_idx)
    
    return matches, unmatched_preds, unmatched_gts

def calculate_confusion_matrix(pred_mask, gt_mask, distance_threshold):
    """
    计算对象级混淆矩阵 (TP, FP, FN)
    
    参数:
        pred_mask: 预测分割掩码 (numpy array)
        gt_mask: 真实分割掩码 (numpy array)
        distance_threshold: 判定为成功匹配的最大距离阈值
    
    返回:
        tp: 真阳性数量 (成功检测)
        fp: 假阳性数量 (错误检测)
        fn: 假阴性数量 (漏检)
    """
    # 对于GT掩码，如果最大值很小，使用自适应阈值
    gt_max = gt_mask.max()
    gt_threshold = 0.5  # 默认阈值
    
    if gt_max < 0.1 and gt_max > 0:  # 最大值小但不为零
        gt_threshold = gt_max / 2
    
    # 确保是二值掩码
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > gt_threshold).astype(np.uint8)
    
    # 提取实例和质心
    pred_instances, pred_centroids = extract_instances(pred_mask)
    gt_instances, gt_centroids = extract_instances(gt_mask)
    
    # 匹配实例
    matches, unmatched_preds, unmatched_gts = match_instances(
        pred_centroids, gt_centroids, distance_threshold
    )
    
    # 计算混淆矩阵
    tp = len(matches)  # 成功匹配数量
    fp = len(unmatched_preds)  # 未匹配的预测实例数量
    fn = len(unmatched_gts)  # 未匹配的真实实例数量
    
    return tp, fp, fn

def calculate_object_metrics(pred_mask, gt_mask, distance_threshold):
    """
    计算对象级检测指标
    
    参数:
        pred_mask: 预测分割掩码 (numpy array)
        gt_mask: 真实分割掩码 (numpy array)
        distance_threshold: 判定为成功匹配的最大距离阈值
        
    返回:
        metrics: 字典，包含各项指标
    """
    # 确保输入是numpy数组
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    
    # 处理多维输入
    if pred_mask.ndim > 2:
        if pred_mask.ndim == 4:  # [B,C,H,W]
            pred_mask = pred_mask[0, 0]
        elif pred_mask.ndim == 3:  # [C,H,W] 或 [B,H,W]
            pred_mask = pred_mask[0]
    
    if gt_mask.ndim > 2:
        if gt_mask.ndim == 4:  # [B,C,H,W]
            gt_mask = gt_mask[0, 0]
        elif gt_mask.ndim == 3:  # [C,H,W] 或 [B,H,W]
            gt_mask = gt_mask[0]
    
    # 计算混淆矩阵
    tp, fp, fn = calculate_confusion_matrix(pred_mask, gt_mask, distance_threshold)
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = fp / (fp + tp) if (fp + tp) > 0 else 0  # FPR = FP / (FP + TP)
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'FPR': false_positive_rate
    }

def evaluate_batch(pred_masks, gt_masks, distance_threshold):
    """
    评估一批预测结果
    
    参数:
        pred_masks: 预测分割掩码批次 [batch_size, height, width]
        gt_masks: 真实分割掩码批次 [batch_size, height, width]
        distance_threshold: 判定为成功匹配的最大距离阈值
        
    返回:
        avg_metrics: 字典，包含平均指标
    """
    batch_metrics = []
    
    for i in range(len(pred_masks)):
        metrics = calculate_object_metrics(pred_masks[i], gt_masks[i], distance_threshold)
        batch_metrics.append(metrics)
    
    # 计算平均指标
    avg_metrics = {}
    for key in batch_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in batch_metrics])
    
    return avg_metrics

class ObjectMetricsCalculator:
    """
    对象级指标计算器，用于在训练或验证过程中累积指标
    """
    def __init__(self, distance_threshold=10):
        self.distance_threshold = distance_threshold
        self.reset()
    
    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.n_samples = 0
    
    def update(self, pred_mask, gt_mask):
        tp, fp, fn = calculate_confusion_matrix(pred_mask, gt_mask, self.distance_threshold)
        
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.n_samples += 1
    
    def compute(self):
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = self.total_fp / (self.total_fp + self.total_tp) if (self.total_fp + self.total_tp) > 0 else 0
        
        return {
            'Object_TP': self.total_tp,
            'Object_FP': self.total_fp,
            'Object_FN': self.total_fn,
            'Object_Precision': precision,
            'Object_Recall': recall,
            'Object_F1': f1_score,
            'Object_FPR': fpr
        }