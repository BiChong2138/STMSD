from typing import Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import copy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace

from data_polyp import get_trainloader,get_testloader
from torch.utils.data import DataLoader
from loss import *
# from models2.refinenet import RefineNet
from torchvision.utils import save_image

output_dir = 'logs'
version_name='Baseline'
logger = TensorBoardLogger(name='vivim_sd',save_dir = output_dir )
import matplotlib.pyplot as plt
# import tent
import math

from medpy import metric
# from misc import *
import misc2
import torchmetrics
from modeling.vivim import Vivim

from poloy_metrics import *
from modeling.utils import JointEdgeSegLoss
from sd_metrics import calculate_object_metrics  # 导入对象级指标计算函数
# torch.set_float32_matmul_precision('high')

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.epochs = self.params.epochs
        self.save_path = os.path.join(os.getcwd(), 'save_images_sd')
        os.makedirs(self.save_path, exist_ok=True)
        self.data_root=self.params.data_root
        self.initlr = self.params.initlr

        self.train_batchsize = self.params.train_bs
        self.val_batchsize = self.params.val_bs

    
        #Train setting
        self.initlr = self.params.initlr #initial learning
        self.weight_decay = self.params.weight_decay #optimizers weight decay
        self.crop_size = self.params.crop_size #random crop size
        self.num_workers = self.params.num_workers
        self.epochs = self.params.epochs
        self.shift_length = self.params.shift_length
        self.val_aug = self.params.val_aug
        self.with_edge = self.params.with_edge

        self.gts = []
        self.preds = []
        
        
        self.nFrames = 5
        self.upscale_factor = 1
        self.data_augmentation = True

        self.criterion = JointEdgeSegLoss(classes=2) if self.with_edge else structure_loss

        self.model = Vivim(with_edge=self.with_edge)


        self.save_hyperparameters()
        

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.999])#,weight_decay=self.weight_decay)
         
        # optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.99],weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)

        return [optimizer], [scheduler]

    # def training_epoch_start(self):
    #     self.scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train')

    def init_weight(self,ckpt_path=None):
        
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
            print(checkpoint.keys())
            checkpoint_model = checkpoint
            state_dict = self.model.state_dict()
            # # 1. filter out unnecessary keys
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            print(checkpoint_model.keys())
            # 2. overwrite entries in the existing state dict
            state_dict.update(checkpoint_model)
            
            self.model.load_state_dict(checkpoint_model, strict=False) 


    def evaluate_one_img(self, pred, gt):


        dice = misc2.dice(pred, gt)
        specificity = misc2.specificity(pred, gt)
        jaccard = misc2.jaccard(pred, gt)
        precision = misc2.precision(pred, gt)
        recall = misc2.recall(pred, gt)
        f_measure = misc2.fscore(pred, gt)

        return dice, specificity, precision, recall, f_measure, jaccard



    def training_step(self, batch, batch_idx):
        self.model.train()

        
        neigbor, target, edge_gt = batch
        # print(edge_gt.shape)
        target = target.cuda()
        #bicubic = bicubic.cuda()
        neigbor = neigbor.cuda()
        bz, nf, nc, h, w = target.shape
        # noisy_images = torch.cat([noisy_images,neigbor_],dim=1)
        # print(neigbor.shape)
        #print("timesteps:",timesteps)#.type())
        if not self.with_edge:
            pred = self.model(neigbor)#, return_dict=False)[0]
            target = target.reshape(bz*nf,nc,h,w)
            loss = self.criterion(pred[self.nFrames//2::self.nFrames], target[self.nFrames//2::self.nFrames])

        else:
            pred,e0 = self.model(neigbor)#, return_dict=False)[0]
            target = target.reshape(bz*nf,nc,h,w)
            edge_gt = edge_gt.reshape(bz*nf,1,h,w)
            loss = self.criterion((pred[self.nFrames//2::self.nFrames], e0[self.nFrames//2::self.nFrames]), (target[self.nFrames//2::self.nFrames], edge_gt[self.nFrames//2::self.nFrames]))

        self.log("train_loss",loss,prog_bar=True)
        # self.log("aux_loss",aux_loss,prog_bar=True)
        return {"loss":loss}


    def on_validation_epoch_end(self):
        # 定义阈值
        Thresholds = np.linspace(1, 0, 256)
        
        # 设置距离阈值
        distance_threshold = 2  # 根据空间碎片大小设置为2像素
        
        # 像素级与对象级指标列表
        dice_lst, jaccard_lst = [], []
        obj_precision_lst, obj_recall_lst, obj_f1_lst, obj_fpr_lst = [], [], [], []
        obj_tp_total, obj_fp_total, obj_fn_total = 0, 0, 0

        for pred, gt in zip(self.preds, self.gts):
            pred = torch.sigmoid(pred)
            
            # 计算更适合的阈值
            gt_max = gt.max()
            
            # 如果gt最大值很小，使用动态阈值
            gt_threshold = 0.5  # 默认阈值
            if gt_max < 0.1:  # 如果最大值小于0.1，使用最大值的一半作为阈值
                gt_threshold = gt_max / 2
            
            # 二值化GT
            gt_binary = (gt > gt_threshold).to(int)
            
            # 像素级指标计算
            dice_l, jaccard_l = [], []
            for threshold in Thresholds:
                pred_one_hot = (pred > threshold).to(int)
                
                # 计算像素级Dice和Jaccard
                dice = misc2.dice(pred_one_hot.detach().cpu().numpy(), gt_binary.detach().cpu().numpy())
                jaccard = misc2.jaccard(pred_one_hot.detach().cpu().numpy(), gt_binary.detach().cpu().numpy())
                
                dice_l.append(dice)
                jaccard_l.append(jaccard)
                
                # 使用阈值0.5计算对象级指标(只计算一次)
                if abs(threshold - 0.5) < 0.01:  # 选择接近0.5的阈值
                    # 计算对象级指标
                    pred_numpy = pred_one_hot.detach().cpu().numpy()
                    gt_numpy = gt_binary.detach().cpu().numpy()
                    
                    obj_metrics = calculate_object_metrics(
                        pred_numpy,  # 预测掩码
                        gt_numpy,    # 真实掩码
                        distance_threshold
                    )
                    
                    # 收集对象级指标
                    obj_precision_lst.append(obj_metrics['Precision'])
                    obj_recall_lst.append(obj_metrics['Recall'])
                    obj_f1_lst.append(obj_metrics['F1-Score'])
                    obj_fpr_lst.append(obj_metrics['FPR'])
                    
                    # 累积TP, FP, FN
                    obj_tp_total += obj_metrics['TP']
                    obj_fp_total += obj_metrics['FP']
                    obj_fn_total += obj_metrics['FN']
            
            # 存储平均像素级指标
            dice_lst.append(sum(dice_l) / len(dice_l))
            jaccard_lst.append(sum(jaccard_l) / len(jaccard_l))

        # 计算平均值
        dice = sum(dice_lst) / len(dice_lst)
        jac = sum(jaccard_lst) / len(jaccard_lst)
        
        # 计算对象级指标平均值
        obj_precision = sum(obj_precision_lst) / len(obj_precision_lst) if obj_precision_lst else 0
        obj_recall = sum(obj_recall_lst) / len(obj_recall_lst) if obj_recall_lst else 0
        obj_f1 = sum(obj_f1_lst) / len(obj_f1_lst) if obj_f1_lst else 0
        obj_fpr = sum(obj_fpr_lst) / len(obj_fpr_lst) if obj_fpr_lst else 0
        
        # 记录指标
        self.log('Dice', dice)
        self.log('Jaccard', jac)
        
        # 记录对象级指标
        self.log('Object_Precision', obj_precision)
        self.log('Object_Recall', obj_recall)
        self.log('Object_F1', obj_f1)
        self.log('Object_FPR', obj_fpr)
        self.log('Object_TP', obj_tp_total)
        self.log('Object_FP', obj_fp_total)
        self.log('Object_FN', obj_fn_total)

        self.gts = []
        self.preds = []
        print("Val: Dice {0}, Jaccard {1}".format(dice, jac))
        print("Object Detection: Precision {0}, Recall {1}, F1 {2}, FPR {3}, TP {4}, FP {5}, FN {6}".format(
            obj_precision, obj_recall, obj_f1, obj_fpr, obj_tp_total, obj_fp_total, obj_fn_total))



    def validation_step(self,batch,batch_idx):
        # torch.set_grad_enabled(True)
        self.model.eval()
        
        neigbor,target = batch
        bz, nf, nc, h, w = neigbor.shape

        # 如果Target最大值很小，使用动态阈值来增强可视化
        if target.max() < 0.1:
            target_viz = target / target.max()  # 归一化到[0,1]范围
        else:
            target_viz = target

        # import time
        # start = time.time()
        if not self.with_edge:
            samples = self.model(neigbor)
        else:
            samples,_ = self.model(neigbor)

        samples = samples[self.nFrames//2::self.nFrames]

        filename = "sample_{}.png".format(batch_idx)
        save_image(samples, os.path.join(self.save_path, filename))      
        filename = "target_{}.png".format(batch_idx)
        save_image(target_viz, os.path.join(self.save_path, filename))
        
        '''
        # 保存二值化后的target
        gt_threshold = target.max() / 2 if target.max() < 0.1 else 0.5
        target_binary = (target > gt_threshold).float()
        filename = "target_binary_{}.png".format(batch_idx)
        save_image(target_binary, os.path.join(self.save_path, filename))
        '''

        self.preds.append(samples)
        self.gts.append(target)
    
    def train_dataloader(self):
        train_loader = get_trainloader(self.data_root, batchsize=self.train_batchsize, trainsize=self.crop_size)
        return train_loader
    
    def val_dataloader(self):
        val_loader = get_testloader(self.data_root, batchsize=self.val_batchsize, trainsize=self.crop_size)
        return val_loader  


def main():
    RESUME = False
    resume_checkpoint_path = r'./checkpoints/ultra-epoch.ckpt'
    if RESUME == False:
        resume_checkpoint_path = None
    
    #128: 32-0.0005
    args={
    'epochs': 200,  #datasetsw
    'data_root':'./SDD/tssnr/SNR500',
    
    'train_bs':4,
    'test_bs':1,
    'val_bs':1, 
    'initlr':1e-4,
    'weight_decay':0.01,
    'crop_size':256,
    'num_workers':4,
    'shift_length':32,
    'val_aug':False,
    'with_edge':False,
    'seed': 1234
    }

    torch.manual_seed(args['seed'])
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.benchmark = True

    hparams = Namespace(**args)

    model = CoolSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
    monitor='Dice',
    dirpath='./checkpointlogs',
    filename='ultra-epoch{epoch:02d}-Dice-{Dice:.4f}-Jaccard-{Jaccard:.4f}',
    auto_insert_metric_name=False,   
    every_n_epochs=1,
    save_top_k=1,
    mode = "max",
    save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        max_epochs=hparams.epochs,
        accelerator='gpu',
        devices=1,
        precision=32,
        logger=logger,
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks = [checkpoint_callback,lr_monitor_callback],
        #accumulate_grad_batches=2  # 添加梯度累积
    ) 

    # 注释掉训练代码
    # trainer.fit(model,ckpt_path=resume_checkpoint_path)
    
    # 指定要验证的模型权重路径
    val_path = r'./checkpointlogs/ultra-epoch184-Dice-0.8364-Jaccard-0.7269.ckpt'  # 请确保这个路径指向您已训练好的模型权重
    # 激活验证代码
    trainer.validate(model, ckpt_path=val_path)
    
if __name__ == '__main__':
	#your code
    main()
