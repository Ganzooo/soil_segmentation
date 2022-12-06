#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import os.path as osp
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from re import I

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import apex
import cv2
from adabelief_pytorch import AdaBelief
from adamp import AdamP
from apex import amp, optimizers
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel as DDP
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from icecream import ic
from skimage import img_as_ubyte
from torch.optim.lr_scheduler import StepLR
from torch_ema import ExponentialMovingAverage
from warmup_scheduler import GradualWarmupScheduler

import options
#Add wanddb
import wandb
from dataset.dataloader import load_dataset
from dataset.dataset_utils import MixUp_AUG
from losses import (CharbonnierLoss, ContentLossWithMask, LossWithMask,
                    bootstrapped_cross_entropy2d)
from metrics import runningScore
from models.DDRNet_23_slim import BasicBlock, DualResNet, DualResNet_imagenet
from models.hardnet_val import hardnet_val
#from models.DDRNet_23 import DualResNet_imagenet, DualResNet, BasicBlock
from models.resnet50_unet_activation import UNetWithResnet50Encoder
from models.resnet50_unet_activation_DUA import UNetWithResnet50EncoderDUA
from utils.image_utils import batch_PSNR, save_img
from utils.utils import get_logger, make_dir

#from torchtools.optim import RangerLars

os.environ["WANDB_API_KEY"] = "0b0a03cb580e75ef44b4dff7f6f16ce9cfa8a290"
#os.environ["WANDB_MODE"] = "dryrun"

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

class Trainer(object):
    def __init__(self, args, logger, writer):
        self.data_path=args.data_path,
        self.batch_size=args.batch_size,
        self.max_epoch=args.max_epoch,
        self.pretrained_weight=args.pretrained_model,
        self.width=args.width,
        self.height=args.height,
        self.resume_train = args.resume,
        self.work_dir = args.work_dir,
        self.n_classes = 3
        self.label_colors = [[0,0,0], [2,2,255],[255,2,2]]
        
        self.train_dataloader, self.val_dataloader = load_dataset(args.data_path, args.batch_size, 
                    distributed=False, center_crop= args.center_crop, random_crop=args.random_crop, resize_size=(args.width,args.height),
                    model_type=args.model_type,color_domain=args.color_domain)
                    
        self.max_epoch = args.max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.width = args.width
        self.height = args.height
        self.args = args
        
        self.best_miou = 0.0
        self.best_test = False

        self.predict_sucess_frame = 0.0
        self.acc_lens_detection = 0.0
        
        # Setup Metrics
        self.running_metrics_val = runningScore(self.n_classes)
        self.running_metrics_val_single = runningScore(self.n_classes)
    
        self.save_model_interval = 1
        self.loss_print_interval = 1
        self.start_epo = 0
        #self.mixup = MixUp_AUG()

        # self.logdir = osp.join(args.work_dir)
        # self.logger = get_logger(self.logdir)
        # self.logger.info("Let the train begin...")
        # self.test_feature_paths = list(sorted(Path(args.data_path).glob("test_input_img/*.png")))

        self.logger = logger
        self.logger.info("Let the train begin...")
        self.writer = writer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ######### Get Model ###########
        if args.model_type == "hardnet":
            self.model = hardnet_val().to(self.device)
        elif args.model_type == "resnet_unet_dua":
            self.model = UNetWithResnet50EncoderDUA().to(self.device)
            ic('ResNet_Unet_DUA model')
        elif args.model_type == "ddrnet":
            self.model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=3, planes=32, spp_planes=128, head_planes=64, augment=False).to(self.device)
            #self.model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=3, planes=64, spp_planes=128, head_planes=128, augment=False).to(self.device)
            ic('Dual ResNet model')
        else:
            self.model = UNetWithResnet50Encoder().to(self.device)
            ic('ResNet_Unet')
        ic('Total parameter of model', sum(p.numel() for p in self.model.parameters()))

        ######### Loss ###########
        if self.args.loss_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        elif self.args.loss_type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean').to(self.device)
        elif self.args.loss_type == 'charb':
            self.criterion = CharbonnierLoss().to(self.device)
        elif self.args.loss_type == 'loss_mask':
            self.criterion = LossWithMask(loss_type='mse').to(self.device)
        elif self.args.loss_type == 'content_loss_mask':
            self.criterion = ContentLossWithMask(loss_type='mse').to(self.device)
        elif self.args.loss_type == 'cross_entropy':
            self.criterion = bootstrapped_cross_entropy2d().to(self.device)
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)

         ######### Load Pretrained model or Resume train ###########
        if args.test_model is not None:
            ic("Testmodel weight load")
            checkpoint = torch.load(args.test_model)
            try:
                self.model.load_state_dict(checkpoint["model_state"])
            except:
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            
        ######### Load Pretrained model or Resume train ###########
        if args.pretrained_model is not None:
            ic("Pretrained weight load")
            checkpoint = torch.load(args.pretrained_model)
            try:
                self.model.load_state_dict(checkpoint["model_state"])
            except:
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

        if args.resume is not None:
            ic("Loading model and optimizer from checkpoint ")
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.start_epo = checkpoint["epoch"]
            self.logger.info("Loaded checkpoint '{}' (epoch {})".format(args.resume, self.start_epo))
            ic(self.start_epo)
        else:
            self.logger.info("No checkpoint found at '{}'".format(args.resume))

        ######### Initialize Optimizer ###########
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        elif args.optimizer == 'adamp':
            self.optimizer = AdamP(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        elif args.optimizer == 'adabelief':
            self.optimizer = AdaBelief(self.model.parameters(), lr=args.lr_initial, eps=1e-16, betas=(0.9, 0.999), weight_decouple = True, rectify = False)
        elif args.optimizer == 'ranger':
            self.optimizer = RangerLars(self.model.parameters(), lr=args.lr_initial)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr_initial, momentum=0.9, weight_decay=args.weight_decay)
        
        ######### Initialize APEX Mixed Prediction ###########
        if args.apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.opt_level)

        ######### Scheduler ###########
        warmup = False
        if warmup:
           warmup_epochs = self.start_epo
           scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch-warmup_epochs, eta_min=1e-6)
           self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
           self.scheduler.step()
        else:
            if args.scheduler == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=1e-6)
            elif args.scheduler == 'cosine_wr':
                self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=args.max_epoch // 3, 
                                                               cycle_mult=1.0, max_lr=args.lr_initial, min_lr=0.000001, 
                                                               warmup_steps=self.max_epoch//12, gamma=0.5)
            elif args.scheduler == 'cosine_wu':
                self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=args.max_epoch, 
                                                               cycle_mult=1.0, max_lr=args.lr_initial, min_lr=0.000001, 
                                                               warmup_steps=self.max_epoch//10, gamma=1.0)
            elif args.scheduler == 'lambda':
                self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** self.max_epoch, last_epoch=-1)
            else:
                self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.1)

            # valid if resume
            for i in range(1, self.start_epo):
                self.scheduler.step()

        ic("==> Training with learning rate:", self.optimizer.param_groups[0]['lr'])
        #ic("==> Training with learning rate:", self.scheduler.get_last_lr()[0])

        ######### Initialize EMA and WandB ###########
        if args.ema:
            self.ema_model = ExponentialMovingAverage(self.model.parameters(), decay=0.995)

        if args.wandb:
            # 1. Start a W&B run
            wandb.init(project='soil_segmentation', entity='gnzrg25',reinit=True, config={"architecture":args.model_type, "dataset": args.data_path,
                    #"scheduler":args.scheduler, "lr_init":self.scheduler.get_last_lr()[0],
                    "scheduler":args.scheduler, "lr_init":self.optimizer.param_groups[0]['lr'],
                    "optim":args.optimizer, "loss":args.loss_type,
                    "batch_size":args.batch_size, "max_epoch":args.max_epoch,
                    "weight_decay":args.weight_decay}) 
    
            run_name = args.work_dir[6:]
            wandb.run.name = run_name
            wandb.run.save()
            # 2. Save model inputs and hyperparameters
            self.config = wandb.config
            #self.config.update(args)

    def step_train(self, mode):
        lr = self.optimizer.param_groups[0]['lr']
        ic('Start {} -> epoch: {}, lr: {}'.format(mode,self.epo, lr))
        self.model.train()

        self.best_test = False
        loss_sum = 0
        iter = 0

        psnr_rgb = []

        tq = tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        tq.set_description('Epoch {}, lr {}'.format(self.epo, lr))
        for index, (img, gt, fname) in tq:
            iter += 1

            img = img.to(self.device)
            gt = gt.to(self.device)

            ### Predict image ###
            pred, pred_id = self.model(img)

            if 'mask' in self.args.loss_type:
                loss = self.criterion(pred, gt, img)
            else:
                loss = self.criterion(pred, gt)

            self.optimizer.zero_grad()
            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            # Update the moving average with the new parameters from the last optimizer step
            if self.args.ema:
                self.ema_model.update()

            loss_sum += loss.item()

            # Dump train images 
            if self.args.save_image_train and np.mod(self.epo, self.args.save_print_interval) == 0:
                if iter < 20:
                    gt = gt.cpu().detach().numpy()
                    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
                    pred_ = pred_id.cpu().detach().numpy()

                    gtRGB = self.decode_segmap(gt[0])
                    predRGB = self.decode_segmap(pred_[0])
                    overlapRGB = img[0] * 0.7 + predRGB * 0.3
                    
                    temp0 = np.concatenate((img[0]*255, gtRGB*255, overlapRGB*255),axis=1)
                    save_img(osp.join(self.args.work_dir, 'result_img','train', str(self.epo) + '/batches_'+ str(iter) + '.jpg'),temp0.astype(np.uint8), color_domain=self.args.color_domain)

            tq.set_postfix(loss='{0:0.4f}'.format(loss_sum / iter))
        tq.close()

        self.scheduler.step()

        ### Tensorboard ###
        avg_loss = loss_sum / iter
        self.writer.add_scalar("Loss/train", avg_loss, self.epo)
        self.writer.add_scalar("Scheduler/lr_value", float(lr), self.epo)
        
        ### Save latest model ###
        if np.mod(self.epo, self.save_model_interval) == 0:
            _state = {
                "epoch": self.epo + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            } 
            _save_path_latest = osp.join(self.args.work_dir,'weight', '{}_latestmodel.pth'.format('soil_segment'))
            torch.save(_state,_save_path_latest)
        
        ### Print result of loss and scheduler###
        format_str = "epoch: {}\n avg loss: {:3f}, scheduler: {}"
        print_str = format_str.format(int(self.epo), float(avg_loss), float(lr))
        ic(print_str)
        self.logger.info(print_str)
        if args.wandb:
            wandb.log({"train/avg loss": avg_loss})
            wandb.log({"train/lr": float(lr)})


    def step_val(self, mode):
        self.model.eval()
        
        ic('Start {} -> epoch: {}'.format(mode,self.epo))
        loss_sum = 0
        iter = 0

        tq = tqdm.tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))
        tq.set_description('Validation')
        for index, (img, gt, fname) in tq:
            iter += 1
            img = img.to(self.device)
            gt = gt.to(self.device)

            with torch.no_grad():
                ###Pred image ###
                pred, pred_id = self.model(img)

                #For tensorboard
                loss = self.criterion(pred, gt)
                loss_sum += loss.item()

                pred_id_ = pred_id.cpu().numpy()
                gt_ = gt.data.cpu().numpy()
                self.running_metrics_val.update(gt_, pred_id_)

            if self.args.save_image_val and np.mod(self.epo, self.args.save_print_interval) == 0:
                gt = gt.cpu().detach().numpy()
                img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
                pred_ = pred_id.cpu().detach().numpy()

                for batch in range(self.args.batch_size-10):
                    gtRGB = self.decode_segmap(gt[batch])
                    predRGB = self.decode_segmap(pred_[batch])
                    overlapRGB = img[batch] * 0.7 + predRGB * 0.3

                    temp0 = np.concatenate((img[batch]*255, gtRGB*255, overlapRGB*255),axis=1)
                    save_img(osp.join(self.args.work_dir, 'result_img','validation',fname[batch]),temp0.astype(np.uint8), color_domain=self.args.color_domain)

            tq.set_postfix(loss='{0:0.4f} '.format(loss_sum/iter))
        tq.close()

        score, class_iou = self.running_metrics_val.get_scores()

        self.running_metrics_val.reset()

        ### Save best model ###
        if score['Mean_IoU'] > self.best_miou:
            self.best_miou = score['Mean_IoU']
            _state = {
                "epoch": index + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            } 

            _save_path = osp.join(self.args.work_dir,'weight', '{}_bestmodel_{}.pth'.format('soil_segment',str(format(float(self.best_miou),'.3f'))))
            directory = os.path.dirname(_save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(_state, _save_path)
        
        ### Print result of PSNR###
        format_str = "epoch: {}\n Overall_Acc: {:3f}, Mean_Acc: {:3f}, Mean_IoU: {:3f}"
        print_str = format_str.format(int(self.epo), float(score['Overall_Acc']), float(score['Mean_Acc']), float(score['Mean_IoU']))
        ic(print_str)
        self.logger.info(print_str)

        ### Tensorboard ###
        self.writer.add_scalar("Val/loss", loss_sum / iter, self.epo)
        for k, v in score.items():
            writer.add_scalar("Val/{}".format(k), v, self.epo + 1)
        for k, v in class_iou.items():
            self.logger.info("{}: {}".format(k, v))
            writer.add_scalar("Val/cls_{}".format(k), v, self.epo + 1)

        if args.wandb:
            wandb.log({"val/avg loss": loss_sum / iter})
            wandb.log({"Val/Overall_Acc": score['Overall_Acc']})
            wandb.log({"Val/Mean_Acc": score['Mean_Acc']})
            wandb.log({"Val/Mean_IoU": score['Mean_IoU']})
            
    def step_test(self, mode):
        self.model.eval()
        
        ic('Start {} -> epoch: {}'.format(mode,0))
        loss_sum = 0
        iter = 0

        try:
            os.remove(osp.join(self.args.work_dir, 'soil_detection_result.csv'))
        except:
            pass
        
        tq = tqdm.tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))
        tq.set_description('Soil Detection')
        for index, (img, gt, fname) in tq:
            iter += 1
            img = img.to(self.device)
            gt = gt.to(self.device)

            with torch.no_grad():
                ###Pred image ###
                pred, pred_id = self.model(img)

                #For tensorboard
                loss = self.criterion(pred, gt)
                loss_sum += loss.item()

                pred_id_ = pred_id.cpu().numpy()
                gt_ = gt.data.cpu().numpy()
                self.running_metrics_val.update(gt_, pred_id_)
                self.running_metrics_val_single.update_single(gt_, pred_id_)

            ### Calculate each frame IoU
            _score_frame, _class_iou_frame = self.running_metrics_val.get_scores()
            _IoU = np.round(_score_frame['Mean_IoU'], 2)
            
            
            if _IoU > 0.5:
                self.predict_sucess_frame = self.predict_sucess_frame + 1    

            for batch in range(1):    
                gt = gt.cpu().detach().numpy()
                img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
                pred_ = pred_id.cpu().detach().numpy()
                gtRGB = self.decode_segmap(gt[batch])
                predRGB = self.decode_segmap(pred_[batch])
                overlapRGB = img[batch] * 0.7 + predRGB * 0.3

                temp0 = np.concatenate((img[batch]*255, gtRGB*255, overlapRGB*255, predRGB*255),axis=1)
                save_img(osp.join(self.args.work_dir, 'result_img','test',fname[batch]),temp0.astype(np.uint8), color_domain=self.args.color_domain)
                    
                #print('Filename: {} \t mIoU: {:.2f}'.format(fname[0], _IoU)) 
                #print('Filename: {} \t Lens status: {} \t Soiled ratio: {}'.format(fname[0], status_lens, float(overlap_ratio)))    
                with open(osp.join(self.args.work_dir, 'soil_detection_result.csv'),'a') as f:
                    f.write('Filename: ,{},mIoU:, {:.2f}\n'.format(fname[0], _IoU))
                    
            tq.set_postfix(Filename='{}'.format(fname[0]), IoU='{0:0.2f}'.format(_IoU))
        tq.close()

        ### Print result of Accuracy###
        print("\n\n")
        print("======== Soiled detection accuracy ========")
        accuracy_of_detection = float(self.predict_sucess_frame / iter)
        format_str = " 오염물질검출정확도: ({})  =  인식에 성공한 프레임수: ({}) / 총 오염물질 이미지 프레임수: ({})"
        print_str = format_str.format(float(accuracy_of_detection), int(self.predict_sucess_frame), int(iter))
        print(print_str)
        
    def step_test_camera_failure(self, mode):
        self.model.eval()
        
        ic('Start {} -> epoch: {}'.format(mode,0))
        loss_sum = 0
        iter = 0

        try:
            os.remove(osp.join(self.args.work_dir, 'camera_failure_gt.csv'))
            with open(osp.join(self.args.work_dir, 'camera_failure_gt.csv'),'w') as f:
                f.write('Filename:,Soiled ratio(GT):, Lens status:\n')
            
            os.remove(osp.join(self.args.work_dir, 'camera_failure_detection_result.csv'))
            with open(osp.join(self.args.work_dir, 'camera_failure_detection_result.csv'),'w') as f:
                f.write('Filename:,Soiled ratio(Pred):, Lens status:\n')
            
        except:
            pass
        
        tq = tqdm.tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))
        tq.set_description('Camera Failure Detection')
        acc_lens_detection_fail = 0
        acc_lens_detection_true = 0
        for index, (img, gt, fname) in tq:
            iter += 1
            img = img.to(self.device)
            gt = gt.to(self.device)

            with torch.no_grad():
                ###Pred image ###
                pred, pred_id = self.model(img)

                #For tensorboard
                loss = self.criterion(pred, gt)
                loss_sum += loss.item()

                pred_id_ = pred_id.cpu().numpy()
                gt_ = gt.data.cpu().numpy()
                self.running_metrics_val.update(gt_, pred_id_)
                self.running_metrics_val_single.update_single(gt_, pred_id_)

            for batch in range(1):
                gt = gt.cpu().detach().numpy()
                img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
                pred_ = pred_id.cpu().detach().numpy()
            
                gtRGB = self.decode_segmap(gt[batch])
                predRGB = self.decode_segmap(pred_[batch])
                overlapRGB = img[batch] * 0.7 + predRGB * 0.3

                nH = gtRGB.shape[0]
                nW = gtRGB.shape[1]
                
                if nW == 1280 and nH == 960: 
                ###Calculate soil ratio from gt
                    ROI_H = [160,800]
                    ROI_W = [160,1120]
                elif nW == 992 and nH == 736:
                    ROI_H = [64,672]
                    ROI_W = [64,928]
                else: 
                    ROI_H = [32,192]
                    ROI_W = [32,288]
                
                _gtR = gtRGB[:,:,1]*255
                _gtR_ROI = _gtR[ROI_H[0]:ROI_H[1],ROI_W[0]:ROI_W[1]]
                calculate_region_of_frame_gt = len(_gtR_ROI[_gtR_ROI>1])
                overlap_ratio_gt = np.round(float(calculate_region_of_frame_gt / (_gtR_ROI.shape[0] * _gtR_ROI.shape[1])) * 100, 2)
                
                ###Calculate soil ratio from prediction
                _predR = predRGB[:,:,1]*255
                _predR_ROI = _predR[ROI_H[0]:ROI_H[1],ROI_W[0]:ROI_W[1]]
                calculate_region_of_frame = len(_predR_ROI[_predR_ROI>1])
                overlap_ratio = np.round(float(calculate_region_of_frame / (_predR_ROI.shape[0] * _predR_ROI.shape[1])) * 100, 2)

                temp0 = np.concatenate((img[batch]*255, overlapRGB*255),axis=1)
                temp0 = cv2.rectangle(temp0,(ROI_W[0],ROI_H[0]),(ROI_W[1],ROI_H[1]), (255,0,0),3)
                temp0 = cv2.rectangle(temp0,(nW+ROI_W[0],ROI_H[0]),(nW+ROI_W[1],ROI_H[1]), (255,0,0),3)
                save_img(osp.join(self.args.work_dir, 'result_img','camera_lens_failure',fname[batch]),temp0.astype(np.uint8), color_domain=self.args.color_domain)
                
                status_type = ['Fail', 'Normal']
                if overlap_ratio > 30:
                    status_lens = 'Fail'
                else: 
                    status_lens = 'Normal'
                    
                if overlap_ratio_gt > 30:
                    status_lens_gt = 'Fail'
                else: 
                    status_lens_gt = 'Normal'
                    
                #print('Filename: {} \t mIoU: {:.2f} \t Soiled ratio(GT): {} \t Soiled ratio(Pred): {} \t Lens status: {}'.format(fname[0], _mIoU, overlap_ratio_gt, overlap_ratio, status_lens)) 
                #print('Filename: {} \t Lens status: {} \t Soiled ratio: {}'.format(fname[0], status_lens, float(overlap_ratio)))    
                
                
                with open(osp.join(self.args.data_path, 'camera_failure_gt.csv'),'a') as f:
                    f.write('{},{}, {} \n'.format(fname[0], overlap_ratio_gt, status_lens_gt))
                    
                with open(osp.join(self.args.work_dir, 'camera_failure_detection_result.csv'),'a') as f:
                    f.write('{},{},{} \n'.format(fname[0], overlap_ratio, status_lens))
                    
                if status_lens_gt == status_lens:
                    self.acc_lens_detection  = self.acc_lens_detection + 1
                
                if status_lens_gt == 'Normal':
                    acc_lens_detection_true += 1
                else: 
                    acc_lens_detection_fail += 1
            #tq.set_postfix(Filename='{}', Soiled_Ratio='{0:0.4f} '.format(fname[0], overlap_ratio))
            tq.set_postfix(Filename='{}'.format(fname[0]), Soiled_ratio='{0:0.2f}'.format(overlap_ratio), Lens_status='{}'.format(status_lens))
        tq.close()

        accuracy_of_detection = float(self.acc_lens_detection / iter)
        
         ### Print result of Accuracy###
        print("\n\n")
        print("======== Camera lens failure detection ========")
        ### Print result of Accuracy###
        format_str = " Camera_lens_status_detection_accuracy: ({})  =  Detection_success_frame: ({}) / Total_frame: ({})"
        print_str = format_str.format(float(accuracy_of_detection), int(self.acc_lens_detection), int(iter))
        print(print_str)
        print("\n\n")
        print("Total lens Failure frame:{} \t Total lens Normal frame:{}".format(acc_lens_detection_true, acc_lens_detection_fail))

    def train(self):
        """Start training."""
        for self.epo in range(self.start_epo, self.max_epoch):
            self.step_train('train')
            self.step_val('val')
        self.writer.close()
        
    def test(self):
        """Start validating."""
        self.step_test('val')
        self.writer.close()
        
    def test_camera_failure(self):
        """Start camera failure test"""
        self.step_test_camera_failure('val')
        self.writer.close()

    def decode_segmap(self, img):
        r = img.copy()
        g = img.copy()
        b = img.copy()
        for l in range(0, self.n_classes):
            r[img == l] = self.label_colors[l][0]
            g[img == l] = self.label_colors[l][1]
            b[img == l] = self.label_colors[l][2]

        rgb = np.zeros((img.shape[0], img.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


if __name__ == "__main__":
    ic.configureOutput(prefix='Soil segmentation training |')
    ######### parser ###########
    args = options.Options().init(argparse.ArgumentParser(description='soil segmentation')).parse_args()
    ic(args)

    ic.enable()
    #ic.disable()

    ##### Tensorboard #####
    
    make_dir(args.work_dir)
    logger = get_logger(args.work_dir + '/log')
    writer = SummaryWriter(log_dir=args.work_dir + '/log')

    trainer = Trainer(args=args, logger=logger, writer=writer)
    #trainer.train()
    
    ###Segment Soil.
    #trainer.test()
    
    ###Detect camera failure.
    trainer.test_camera_failure()
