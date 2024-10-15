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
from models.GAN_Template_py import GAN_Template

# from torch.autograd import Variable
from torch.utils.data import DataLoader
import piq
from tqdm import tqdm


class UDCTS_GAN(GAN_Template):
    def __init__(self):
        super(UDCTS_GAN,self).__init__()
        self.generator = UDCTS_Generator()
        self.discriminator = UDCTS_Discriminator()

        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.bsl1_loss_module = BrightnessDarknessLossL1(alpha=1/5,y_black_factor=1,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        self.bsl2_loss_module = BrightnessDarknessLossL2(alpha=1/5,y_black_factor=1,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        
        self.s1_bsl1_loss_module = BrightnessDarknessLossL1(alpha=1/8,y_black_factor=4,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        self.s1_bsl2_loss_module = BrightnessDarknessLossL2(alpha=1/8,y_black_factor=4,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        
        self.s2_bsl1_loss_module = BrightnessDarknessLossL1(alpha=1/8,y_black_factor=1,y_gray_factor=1,y_white_factor=4,y_brightness_factor=1,x_black_factor=1)
        self.s2_bsl2_loss_module = BrightnessDarknessLossL2(alpha=1/8,y_black_factor=1,y_gray_factor=1,y_white_factor=4,y_brightness_factor=1,x_black_factor=1)
        
        self.s3_bsl1_loss_module = BrightnessDarknessLossL1(alpha=1/5,y_black_factor=1,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        self.s3_bsl2_loss_module = BrightnessDarknessLossL2(alpha=1/5,y_black_factor=1,y_gray_factor=1,y_white_factor=1,y_brightness_factor=1,x_black_factor=1)
        
        self.mse_loss_module = torch.nn.MSELoss()

        
    def init_config_dataset(self):
        # config = self.config
        dataset_name = self.config.opt.dataset_name
        transforms_ = self.transforms_
        batch_size = self.config.opt.batch_size
        num_workers = self.config.opt.num_workers
        
        
        
        self.dataloader_train = DataLoader(
        FromToDataset(dataset_name,transforms_=transforms_,to_a='A-train',from_b='B-train',config=self.config),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.train_data_length = len(self.dataloader_train)
        
        self.dataloader_valid = DataLoader(
            FromToDataset(dataset_name,transforms_=transforms_,to_a='A-test',from_b='B-test',config=self.config),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.valid_data_length = len(self.dataloader_valid)
        
        # epoch_start = self.config.opt.epoch_start
        # epoch_end = self.config.opt.epoch_end
        # train_csv_path = os.path.join(self.dir,'train_log.csv')
        # # self.logger = CSVLogger(epoch_end, self.train_data_length ,train_csv_path)
        
    def init_config_optimizer(self):
        # Optimizers
        opt = self.config.opt
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        
    def compute_loss_D(self,y_pred,y_truth,x_input):
        self.loss_D = None
        patch = self.patch
        valid = torch.tensor(np.ones((y_truth.size(0), *patch)), dtype=torch.float32, device='cuda', requires_grad=False)
        fake = torch.tensor(np.zeros((y_truth.size(0), *patch)), dtype=torch.float32, device='cuda', requires_grad=False)
        
        pred_real = self.discriminator(x_input,y_truth)
        pred_fake = self.discriminator(x_input,y_pred.detach())
        
        loss_real_value = self.mse_loss_module(pred_real,valid)
        loss_fake_value = self.mse_loss_module(pred_fake, fake)
        self.loss_D = 0.5 * (loss_real_value + loss_fake_value)
        # return self.loss_D
    
    def compute_loss_G(self, y_pred, y_truth, x_input,stage1,stage2):
        self.loss_G = None
        
        normal_l1loss_value = self.mse_loss_module(y_pred,y_truth)
        
        s1_l1_loss = self.s1_bsl1_loss_module(stage1,y_truth,x_input)
        s1_l2_loss = self.s1_bsl2_loss_module(stage1,y_truth,x_input)
        
        s2_l1_loss = self.s2_bsl1_loss_module(stage2,y_truth,x_input)
        s2_l2_loss = self.s2_bsl2_loss_module(stage2,y_truth,x_input)
        
        s3_l1_loss = self.s3_bsl1_loss_module(y_pred,y_truth,x_input)
        s3_l2_loss = self.s3_bsl2_loss_module(y_pred,y_truth,x_input)
        
        self.loss_G = normal_l1loss_value + s1_l1_loss + s1_l2_loss + s2_l1_loss + s2_l2_loss + s3_l1_loss + s3_l2_loss
        # return self.loss_G
        
    def compute_loss(self, y_pred, y_truth, x_input,stage1,stage2):
        
        # loss_D phase
        self.loss_D = None
        patch = self.patch
        valid = torch.tensor(np.ones((y_truth.size(0), *patch)), dtype=torch.float32, device=self.device, requires_grad=False)
        fake = torch.tensor(np.zeros((y_truth.size(0), *patch)), dtype=torch.float32, device=self.device, requires_grad=False)
        
        pred_real = self.discriminator(x_input,y_truth)
        pred_fake = self.discriminator(x_input,y_pred)
        
        loss_real_value = self.mse_loss_module(pred_real,valid)
        loss_fake_value = self.mse_loss_module(pred_fake, fake)
        self.loss_D = 0.5 * (loss_real_value + loss_fake_value)
        
        # loss_G phase
        self.loss_G = None
        
        
        loss_G_ad = self.mse_loss_module(pred_fake, valid)
        
        normal_l1loss_value = self.mse_loss_module(y_pred,y_truth)
        
        s1_l1_loss = self.s1_bsl1_loss_module(stage1,y_truth,x_input)
        s1_l2_loss = self.s1_bsl2_loss_module(stage1,y_truth,x_input)
        
        s2_l1_loss = self.s2_bsl1_loss_module(stage2,y_truth,x_input)
        s2_l2_loss = self.s2_bsl2_loss_module(stage2,y_truth,x_input)
        
        s3_l1_loss = self.s3_bsl1_loss_module(y_pred,y_truth,x_input)
        s3_l2_loss = self.s3_bsl2_loss_module(y_pred,y_truth,x_input)
        
        self.loss_G = loss_G_ad + (normal_l1loss_value + s1_l1_loss + s1_l2_loss + s2_l1_loss + s2_l2_loss + s3_l1_loss + s3_l2_loss) / 7

    