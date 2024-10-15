import os
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import datetime
from back.GAN_py import GAN_base
from back.Base_Method_py import de_norm,limit_zero2one
from back.Base_Dataset_py import FromToDataset
from back.Base_Loss_py import BrightnessDarknessLossL1,BrightnessDarknessLossL2
from models.G_UDCTS_Generator_py import UDCTS_Generator
from models.D_UDCTS_Discriminator_py import UDCTS_Discriminator
from models.DenseDoubleUnetModel_py import DenseDoubleUnet_G
from models.G_L7_Generator_py import L7_UpwardDense_G
from models.G_Template_UpwardDense_py import G_Template_UpwardDense
# from torch.autograd import Variable
from torch.utils.data import DataLoader
import piq
from tqdm import tqdm
from models.GAN_Template_py import GAN_Template


class GAN_UpwardDense(GAN_Template):
    def __init__(self):
        super(GAN_UpwardDense,self).__init__()
        # self.generator = UDCTS_Generator()
        self.generator = G_Template_UpwardDense()
        self.discriminator = UDCTS_Discriminator()

        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
    
        self.l2_loss_module_G = torch.nn.MSELoss()
        self.l1_loss_module_G = torch.nn.L1Loss()
        self.mse_loss_module_D = torch.nn.MSELoss()
        
        self.bsl1_loss_module = BrightnessDarknessLossL1(alpha=1/5,y_black_factor=1,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        self.bsl2_loss_module = BrightnessDarknessLossL2(alpha=1/5,y_black_factor=1,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        
        self.s1_bsl1_loss_module = BrightnessDarknessLossL1(alpha=1/8,y_black_factor=4,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        self.s1_bsl2_loss_module = BrightnessDarknessLossL2(alpha=1/8,y_black_factor=4,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        
        self.s2_bsl1_loss_module = BrightnessDarknessLossL1(alpha=1/8,y_black_factor=1,y_gray_factor=1,y_white_factor=4,y_brightness_factor=1,x_black_factor=1)
        self.s2_bsl2_loss_module = BrightnessDarknessLossL2(alpha=1/8,y_black_factor=1,y_gray_factor=1,y_white_factor=4,y_brightness_factor=1,x_black_factor=1)
        
        self.s3_bsl1_loss_module = BrightnessDarknessLossL1(alpha=1/5,y_black_factor=1,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        self.s3_bsl2_loss_module = BrightnessDarknessLossL2(alpha=1/5,y_black_factor=1,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        

        
    def compute_loss(self, y_pred, y_truth, x_input,stage1,stage2):
        
        # loss_D phase
        self.loss_D = None
        patch = self.patch
        valid = torch.tensor(np.ones((y_truth.size(0), *patch)), dtype=torch.float32, device=self.device, requires_grad=False)
        fake = torch.tensor(np.zeros((y_truth.size(0), *patch)), dtype=torch.float32, device=self.device, requires_grad=False)
        
        pred_real = self.discriminator(y_truth,x_input)
        pred_fake = self.discriminator(y_pred,x_input)
        
        loss_real_value_D = self.mse_loss_module(pred_real,valid)
        loss_fake_value_D = self.mse_loss_module(pred_fake, fake)
        self.loss_D = 0.5 * (loss_real_value_D + loss_fake_value_D)
        
        # loss_G phase
        self.loss_G = None
        
        loss_G_ad = self.mse_loss_module(pred_fake, valid)
        loss_G_l1 = self.l1_loss_module_G(y_pred,y_truth)
        loss_G_l2 = self.l2_loss_module_G(y_pred,y_truth)
        
 
        # loss_G phase
        self.loss_G = None
        
        
        loss_G_ad = self.mse_loss_module(pred_fake, valid)
        
        s1_l1_loss = self.s1_bsl1_loss_module(stage1,y_truth,x_input)
        s1_l2_loss = self.s1_bsl2_loss_module(stage1,y_truth,x_input)
        
        s2_l1_loss = self.s2_bsl1_loss_module(stage2,y_truth,x_input)
        s2_l2_loss = self.s2_bsl2_loss_module(stage2,y_truth,x_input)
        
        s3_l1_loss = self.s3_bsl1_loss_module(y_pred,y_truth,x_input)
        s3_l2_loss = self.s3_bsl2_loss_module(y_pred,y_truth,x_input)
        
        self.loss_G = loss_G_ad + (loss_G_l1 +loss_G_l2 + s1_l1_loss + s1_l2_loss + s2_l1_loss + s2_l2_loss + s3_l1_loss + s3_l2_loss)/8
        

    
