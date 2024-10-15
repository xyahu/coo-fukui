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
# from torch.autograd import Variable
from torch.utils.data import DataLoader
import piq
from tqdm import tqdm


class GAN_Template(GAN_base):
    def __init__(self):
        super(GAN_Template,self).__init__()
        # self.generator = UDCTS_Generator()
        self.generator = DenseDoubleUnet_G()
        self.discriminator = UDCTS_Discriminator()

        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
       
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
        
        
    def compute_loss(self, y_pred, y_truth, x_input,stage1,stage2):
        
        # loss_D phase
        # self.loss_D = None
        patch = self.patch
        valid = torch.tensor(np.ones((y_truth.size(0), *patch)), dtype=torch.float32, device=self.device, requires_grad=False)
        fake = torch.tensor(np.zeros((y_truth.size(0), *patch)), dtype=torch.float32, device=self.device, requires_grad=False)
        
        pred_real = self.discriminator(y_truth,x_input)
        pred_fake = self.discriminator(y_pred,x_input)
        
        loss_real_value_D = self.mse_loss_module(pred_real,valid)
        loss_fake_value_D = self.mse_loss_module(pred_fake, fake)
        self.loss_D = 0.5 * (loss_real_value_D + loss_fake_value_D)
        
        # loss_G phase
        # self.loss_G = None
        
        loss_G_ad = self.mse_loss_module(pred_fake, valid)
        loss_G_l1 = self.l1_loss_module_G(y_pred,y_truth)
        loss_G_l2 = self.l2_loss_module_G(y_pred,y_truth)
        
        self.loss_G = loss_G_ad + loss_G_l1 +loss_G_l2
    
    def expand_tensors_1to3(self, *tensors):
        return [tensor.expand(-1, 3, -1, -1) for tensor in tensors]
    
    def limit_tensors_0to1(self,*tensors):
        return [limit_zero2one(tensor) for tensor in tensors]

    def compute_and_accumulate_metrics(self, y_pred, y_truth, metrics_total):
        ssim, ms_ssim, psnr, mse, mae = self.calculate_evaluation_index(y_pred, y_truth)
        metrics_total['ssim'] += ssim
        metrics_total['ms_ssim'] += ms_ssim
        metrics_total['psnr'] += psnr
        metrics_total['mse'] += mse
        metrics_total['mae'] += mae
        return ssim, ms_ssim, psnr, mse, mae

    def initialize_metrics(self):
        return {'loss_G': 0, 'loss_D': 0, 'ssim': 0, 'ms_ssim': 0, 'psnr': 0, 'mse': 0, 'mae': 0}



    def train_s(self):
         # Tensor type
        Tensor = self.Tensor
        prev_time = time.time()
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.config.opt.epoch_start,self.config.opt.epoch_end):
            metrics_total = self.initialize_metrics()
            loop_train = tqdm(enumerate(self.dataloader_train), total=self.train_data_length, colour='green', ncols=self.config.cmd_bytes)
            for i, batch in loop_train:
                # Model inputs
                y_truth = torch.autograd.Variable(batch["A"].type(Tensor)) # To A
                x_input = torch.autograd.Variable(batch["B"].type(Tensor)) # From B 
                
                self.optimizer_G.zero_grad()
                self.optimizer_D.zero_grad()
                
                # inference
                y_pred, y_stage2, y_stage1 = self.generator(x_input)
                # expand channel from 1 to 3, and limit all var into 0 and 1
                y_pred, y_stage2, y_stage1 = self.expand_tensors_1to3(y_pred, y_stage2, y_stage1)
                # y_truth = limit_zero2one(y_truth)
                
                # calculate_loss
                self.compute_loss(y_pred=y_pred, y_truth=y_truth, x_input=x_input, stage1=y_stage1, stage2=y_stage2)
                
                # Determine approximate time left
                batches_done = epoch * self.train_data_length + i + 1
                self.batches_done = batches_done
                # batches_left = self.config.opt.epoch_start * self.train_data_length - batches_done
                # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                # prev_time = time.time()
                
                ssim, ms_ssim, psnr, mse, mae = self.compute_and_accumulate_metrics(y_pred, y_truth, metrics_total)
                metrics_total['loss_G'] += self.loss_G.item()
                metrics_total['loss_D'] += self.loss_D.item()

                # loop_train.set_postfix(ssim=ssim,psnr=psnr,mse=mse,mae=mae)
                loop_train.set_postfix_str(f"loss_G={self.loss_G:.4f},loss_D={self.loss_D:.4f},ssim={ssim:.4f},ms_ssim={ms_ssim:.4f},psnr={psnr:.4f},mse:{mse:.4f},mae:{mae:.4f},[{int(batches_done / self.train_data_length):03d}/{self.config.opt.epoch_end:03d}]")
                
                train_images = {'x_input':x_input,
                                'y_truth':y_truth,
                                'y_pred':y_pred,
                                'y_stage2':y_stage2,
                                'y_stage1':y_stage1}
                
                
                # 这里可以设置是否启用web服务（Visdom）
                # self.logger.log(losses=train_indexs, images=train_images)
                
                
                # 合并损失
                # total_loss = self.loss_G + self.loss_D
                # total_loss.backward()
                
                self.loss_G.backward(retain_graph=True)
                self.loss_D.backward()
                self.optimizer_G.step()
                self.optimizer_D.step()
                
                if batches_done % self.train_data_length == 0:
                    self.sample_tensors_as_images(train_images,batches_done,'train')
            
            self.epoch_done = batches_done / self.train_data_length
            number = self.train_data_length
            
            avg_metrics = {k: v / number for k, v in metrics_total.items()}
            train_avg_indexs = {'Epoch': self.epoch_done}
            train_avg_indexs.update({k: f"{v:.4f}" for k, v in avg_metrics.items()})
            
            self.save_dict_as_csv(train_avg_indexs,'log_train.csv')
            # 1 train loop end
            
            if batches_done % (self.train_data_length * self.config.opt.sample_interval_valid)== 0:
                self.valid_loop()
                
                
            if epoch % self.config.opt.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

    def valid_loop(self,save_image_flag=True):
        Tensor = self.Tensor
        # self.generator.eval()
        # self.discriminator.eval()
        with torch.no_grad():
            metrics_total = self.initialize_metrics()
            loop = tqdm(enumerate(self.dataloader_valid),total=self.valid_data_length,colour='white',ncols=self.config.cmd_bytes)
            
            if save_image_flag:
                valid_path = os.path.join(self.sample_path, 'valid')
                epoch_str = f"{int(self.epoch_done):03d}"
                dir_names = ["x_input", "y_truth", "y_pred", "y_stage2", "y_stage1", "combined"]
                dir_paths = {name: os.path.join(valid_path, epoch_str, name) for name in dir_names}
                for path in dir_paths.values():
                    os.makedirs(path, exist_ok=True)
            
            for i, batch in loop:
            # for i, batch in enumerate(self.dataloader_valid):
                # Model inputs
                y_truth = torch.autograd.Variable(batch["A"].type(Tensor)) # To A
                x_input = torch.autograd.Variable(batch["B"].type(Tensor)) # From B 
                
                # inference
                y_pred, y_stage2, y_stage1 = self.generator(x_input)
                y_pred, y_stage2, y_stage1 = self.expand_tensors_1to3(y_pred, y_stage2, y_stage1)
                
                # 提前使用limit会使保存的图像变灰。同时也会降低指标
                # y_truth = limit_zero2one(y_truth)
                
                # calculate_loss
                self.compute_loss(y_pred=y_pred, y_truth=y_truth, x_input=x_input, stage1=y_stage1, stage2=y_stage2)
                metrics_total['loss_G'] += self.loss_G.item()
                metrics_total['loss_D'] += self.loss_D.item()

                # loop_train.set_postfix(ssim=ssim,psnr=psnr,mse=mse,mae=mae)
               
                ssim, ms_ssim, psnr, mse, mae = self.compute_and_accumulate_metrics(y_pred, y_truth, metrics_total)
                # ssim,ms_ssim,psnr,mse,mae = ssim.item(),ms_ssim.item(),psnr.item(),mse.item(),mae.item()
                loop.set_postfix_str(f"loss_G={self.loss_G:.4f},loss_D={self.loss_D:.4f},ssim={ssim:.4f},ms_ssim={ms_ssim:.4f},psnr={psnr:.4f},mse:{mse:.4f},mae:{mae:.4f},[{int(self.epoch_done):03d}/{self.config.opt.epoch_end:03d}]")
                
                # loop.set_postfix_str(
                #     f"ssim={ssim:.4f},ms_ssim={ms_ssim:.4f},psnr={psnr:.4f},mse:{mse:.4f},mae:{mae:.4f},[{int(self.epoch_done):03d}/{self.config.opt.epoch_end:03d}]"
                # )


                if save_image_flag:
                    valid_images = {
                        'x_input': x_input,
                        'y_truth': y_truth,
                        'y_pred': y_pred,
                        'y_stage2': y_stage2,
                        'y_stage1': y_stage1
                    }            
                    self.sample_tensors_as_images(valid_images,i,'valid')


            """ 保存各项指标 """
            number = self.valid_data_length
            avg_metrics = {k: v / number for k, v in metrics_total.items()}
            valid_results = {'Epoch': self.epoch_done}
            valid_results.update({k: f"{v:.4f}" for k, v in avg_metrics.items()})
            self.save_dict_as_csv(valid_results, 'log_evalu.csv')
            
        # self.generator.train()
        # self.discriminator.train()     
                
    def sample_tensors_as_images(self,images,idx,flag='train'):
        dir_this = os.path.join(self.sample_path,flag)
        epoch_str = f"{int(self.epoch_done):03d}"
        dir_this = os.path.join(dir_this,epoch_str)
        dir_combined = os.path.join(dir_this,'combined')
        os.makedirs(dir_combined, exist_ok=True)
        combined_image = None

        for image_name, tensor in images.items():  
            dir_image_name = os.path.join(dir_this,image_name)
            os.makedirs(dir_image_name, exist_ok=True)
            
            # Normalize images to a [0, 1] scale
            dn_image = de_norm(tensor.data)
            dn_image = dn_image.expand(-1, 3, -1, -1) 
            
            save_path = os.path.join(dir_image_name, f"{image_name}_{idx:08d}.png")
            
            # Save each image in the folder respectively.
            torchvision.utils.save_image(dn_image, save_path)

            # Stack all images vertically
            if combined_image is None:
                combined_image = dn_image
            else:
                combined_image = torch.cat([combined_image, dn_image], dim=2)
            
        # Save the combined image
        save_path = os.path.join(dir_combined, f"combined_{idx:08d}.png")
        # Save all images in one column
        torchvision.utils.save_image(combined_image, save_path)
        pass
                
                
                
    
    def calculate_evaluation_index(self,y_pred,y_truth,need_limit=1):
        # ssim = piq.ssim(limit_y_pred, limit_y_truth, data_range=1.).cpu()
        # ms_ssim = piq.multi_scale_ssim(limit_y_pred, limit_y_truth, data_range=1.).cpu()
        # psnr = piq.psnr(limit_y_pred, limit_y_truth, data_range=1., reduction='none').cpu()
        # # vif = piq.vif_p(limit_y_truth, limit_y_pred, data_range=1.).cpu()
        # mse = (torch.abs(limit_y_pred - limit_y_truth) ** 2).mean() # L2
        # mae = torch.abs(limit_y_pred - limit_y_truth).mean() # L1
        
        y_fake = y_pred.clone()
        y_real = y_truth.clone()
        if need_limit == 1:
            limit_y_pred  = limit_zero2one(y_fake)
            limit_y_truth = limit_zero2one(y_real)
            
        
        ssim = piq.ssim(limit_y_pred, limit_y_truth, data_range=1.)
        ms_ssim = piq.multi_scale_ssim(limit_y_pred, limit_y_truth, data_range=1.)
        psnr = piq.psnr(limit_y_pred, limit_y_truth, data_range=1., reduction='none')
        # vif = piq.vif_p(limit_y_truth, limit_y_pred, data_range=1.).cpu()
        mse = (torch.abs(limit_y_pred - limit_y_truth) ** 2).mean() # L2
        mae = torch.abs(limit_y_pred - limit_y_truth).mean() # L1
        
        return ssim.item(),ms_ssim.item(),psnr.item(),mse.item(),mae.item()
        # ssim,ms_ssim,psnr,mse,mae = ssim.item(),ms_ssim.item(),psnr.item(),mse.item(),mae.item()
        # return ssim,ms_ssim,psnr,mse,mae
