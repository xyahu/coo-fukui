import os
import torch
# import numpy as np
from torch import nn
from torchvision.utils import save_image
from back.CVS_Module_py import CVS_Module
from torch.utils.data import DataLoader
from back.Base_Dataset_py import FromToDataset
import csv
import time
from tqdm import tqdm
import piq


class Base_Model_Vector(CVS_Module):
    def __init__(self,c,m):
        super(Base_Model_Vector, self).__init__()
        self.config = c
        self.model = m
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.init(self.config)
    
    def init(self,config):
        super(Base_Model_Vector, self).init(config)
        
        classname = self.__class__.__name__
        if config.opt.pth is not None:
            pth = config.opt.pth
            if os.path.exists(pth):
                print(f"{classname} 指定的存档文件存在,正在读取：{pth}")
                # Load pretrained models
                self.generator.load_state_dict(torch.load(pth))
                print(f"存档读取成功...")
                
                s_epoch = pth.split("_")[-1]
                s_epoch = s_epoch.split(".")[0]
                pth_epoch = int(s_epoch)
                self.config.opt.epoch_start = pth_epoch
            else:
                raise("存档文件不存在，请重新检查")
            pass
        else:
            # Initialize weights
            self.model.apply(self.weights_init_normal)
            # self.discriminator.apply(self.weights_init_normal)
        
        print(f"{classname}.init() done.")
        
    def init_config_dataset(self,A_train='A-train',B_train='B-train',A_val='A-val',B_val='B-val'):
        # config = self.config
        dataset_name = self.config.opt.dataset_name
        transforms_ = self.transforms_
        batch_size = self.config.opt.batch_size
        num_workers = self.config.opt.num_workers
        
        
        
        self.dataloader_train = DataLoader(
        FromToDataset(dataset_name,transforms_=transforms_,to_a=A_train,from_b=B_train,config=self.config),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.train_data_length = len(self.dataloader_train)
        
        self.dataloader_valid = DataLoader(
            FromToDataset(dataset_name,transforms_=transforms_,to_a=A_val,from_b=B_val,config=self.config),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.valid_data_length = len(self.dataloader_valid)
        
        # from back.Base_Dataset_py import NormalImageDataset
        # self.dataloader_valid_2 = DataLoader(
        #     NormalImageDataset(dataset_name,transforms_=transforms_,folder_name='total-val',config=self.config),
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=num_workers,
        # )
        # self.valid_data_length_2 = len(self.dataloader_valid_2)
        
        
    # def adapt_cuda(self):
    #     # Assuming model is your neural network model
    #     if self.device is not torch.device('cpu'):
    #         m = m.cuda(self.device)  # Move model to GPU
    #         if len(self.device_ids) > 1:
    #             # Use DataParallel to utilize multiple GPUs
    #             print("Use DataParallel to utilize multiple GPUs")
    #             m = torch.nn.DataParallel(m, device_ids=self.device_ids)
    #     else:
    #         # No GPU available, keep the model on CPU
    #         print("No GPU available, keep the model on CPU")
    #         pass  # Model is already on CPU by default
    
    def initialize_metrics(self):
        return {'loss': 0,'correct':0,'total':0}
    
    def calculate_evaluation_index(self,y_pred,y_truth,need_limit=1):
        # ssim = piq.ssim(limit_y_pred, limit_y_truth, data_range=1.).cpu()
        # ms_ssim = piq.multi_scale_ssim(limit_y_pred, limit_y_truth, data_range=1.).cpu()
        # psnr = piq.psnr(limit_y_pred, limit_y_truth, data_range=1., reduction='none').cpu()
        # # vif = piq.vif_p(limit_y_truth, limit_y_pred, data_range=1.).cpu()
        # mse = (torch.abs(limit_y_pred - limit_y_truth) ** 2).mean() # L2
        # mae = torch.abs(limit_y_pred - limit_y_truth).mean() # L1
        from back.Base_Method_py import limit_zero2one
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

    def compute_and_accumulate_metrics(self, y_pred, y_truth, metrics_total):
        ssim, ms_ssim, psnr, mse, mae = self.calculate_evaluation_index(y_pred, y_truth)
        metrics_total['ssim'] += ssim
        metrics_total['ms_ssim'] += ms_ssim
        metrics_total['psnr'] += psnr
        metrics_total['mse'] += mse
        metrics_total['mae'] += mae
        return ssim, ms_ssim, psnr, mse, mae
    
    def train_s(self):
         # Tensor type
        Tensor = self.Tensor
        prev_time = time.time()
        self.model.train()
        for epoch in range(self.config.opt.epoch_start,self.config.opt.epoch_end):
            metrics_total = self.initialize_metrics()
            loop_train = tqdm(enumerate(self.dataloader_train), total=self.train_data_length, colour='green', ncols=self.config.cmd_bytes)
            for i, batch in loop_train:
                # Model inputs
                img_A_tensor = batch["A"].type(Tensor) # To A
                img_B_tensor = batch["B"].type(Tensor) # From B 
                # img_B_tensor = torch.autograd.Variable(batch["B"].type(Tensor)) # From B 
                # COMPLETE THIS
                # img_A_tensor 的标签值应为-1,请创建相应的标签值
                # img_B_tensor 的标签值应为1,请创建相应的标签值
                # 计算损失函数的方法是self.compute_loss(),可以直接调用，补充完成后续的代码
                
                # 创建标签
                # img_A_tensor 的标签值应为 -1
                # device = img_B_tensor.device
                dtype = img_A_tensor.dtype
                # print(f'\ndtype:{type(dtype)}\nvalue:{dtype}')
                labels_A = -torch.ones((img_A_tensor.size(0)), dtype=dtype, device=self.device)

                # img_B_tensor 的标签值应为 1
                labels_B =  torch.ones((img_A_tensor.size(0)), dtype=dtype, device=self.device)
                # labels_B = Tensor(img_B_tensor.size(0)).fill_(1)
                
                # 合并输入和标签
                inputs = torch.cat((img_A_tensor, img_B_tensor), dim=0)
                labels = torch.cat((labels_A, labels_B), dim=0)
                
                
                # inference
                # 前向传播
                outputs = self.model(inputs)

                # 计算损失函数
                loss = self.compute_loss(outputs, labels)
                
                 # 计算预测值
                # 假设模型输出为 [batch_size, 1]，需要调整形状
                outputs = outputs.view(-1)
                preds = torch.sign(outputs)
                
                 # 计算准确率
                correct = (preds == labels).sum().item()
                total = labels.size(0)

                # print(f'train stage : accuracy:{accuracy}')
                # time.sleep(0.5)
                
                # 更新指标
                metrics_total['loss'] += loss.item()
                metrics_total['correct'] += correct
                metrics_total['total'] += total
                
                # 将张量转换为列表,并格式化
                outputs_value = outputs.tolist()
                labels_value = labels.tolist()
                formatted_outputs_value =[float(f'{num:.2f}') for num in outputs_value]
                formatted_accuracy_value = float(f'{(metrics_total['correct']/metrics_total['total']):.4f}')

                # # 使用 map 和 lambda 函数格式化输出
                # formatted_outputs_value = [list(map(lambda x: f"{x:.2f}", row)) for row in outputs_value]
                # formatted_labels_value = [list(map(lambda x: f"{x:.2f}", row)) for row in labels_value]
                torch.set_printoptions(precision=2)

                # 更新进度条信息
                loop_train.set_description(f"Train Epoch [{epoch}/{self.config.opt.epoch_end}]")
                loop_train.set_postfix(loss=loss.item(),correct=metrics_total['correct'],total=metrics_total['total'],accuracy=formatted_accuracy_value)
                
                # Determine approximate time left
                batches_done = epoch * self.train_data_length + i + 1
                self.batches_done = batches_done
                # batches_left = self.config.opt.epoch_start * self.train_data_length - batches_done
                # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                # prev_time = time.time()
                
                
                # 这里可以设置是否启用web服务（Visdom）
                # self.logger.log(losses=train_indexs, images=train_images)
                
                
                # 合并损失
                # total_loss = self.loss_G + self.loss_D
                # total_loss.backward()
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # 计算并打印当前 epoch 的平均损失和准确率
            avg_loss = metrics_total['loss'] / self.train_data_length
            avg_accuracy = metrics_total['correct'] / metrics_total['total']
            # print(f"Epoch [{epoch}/{self.config.opt.epoch_end}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            result_avg = {'epoch':epoch,'loss':avg_loss,'accuracy':avg_accuracy}
            self.save_dict_as_csv(result_avg,'log_train.csv')
                # if batches_done % self.train_data_length == 0:
                #     self.sample_tensors_as_images(train_images,batches_done,'train')
            
            self.epoch_done = int(batches_done / self.train_data_length)
            # number = self.train_data_length
            
            # avg_metrics = {k: v / number for k, v in metrics_total.items()}
            # train_avg_indexs = {'Epoch': self.epoch_done}
            # train_avg_indexs.update({k: f"{v:.4f}" for k, v in avg_metrics.items()})
            
            # self.save_dict_as_csv(train_avg_indexs,'log_train.csv')
            # # 1 train loop end
            
            if batches_done % (self.train_data_length * self.config.opt.sample_interval_valid)== 0:
                self.valid_loop()
                
                
            if epoch % self.config.opt.checkpoint_interval == 0:
                self.save_checkpoint(epoch)
        pass
    
    def valid_loop(self):
         # Tensor type
        Tensor = self.Tensor
        prev_time = time.time()
        self.model.eval()
        
        metrics_total = self.initialize_metrics()
        loop_valid = tqdm(enumerate(self.dataloader_valid), total=self.valid_data_length, colour='yellow', ncols=self.config.cmd_bytes)
        for i, batch in loop_valid:
            # Model inputs
            img_A_tensor = batch["A"].type(Tensor) # To A
            img_B_tensor = batch["B"].type(Tensor) # From B 
            # img_B_tensor = torch.autograd.Variable(batch["B"].type(Tensor)) # From B 
            # COMPLETE THIS
            # img_A_tensor 的标签值应为-1,请创建相应的标签值
            # img_B_tensor 的标签值应为1,请创建相应的标签值
            # 计算损失函数的方法是self.compute_loss(),可以直接调用，补充完成后续的代码
            
            # 创建标签
            # img_A_tensor 的标签值应为 -1
            dtype = img_A_tensor.dtype
            labels_A = -torch.ones((img_A_tensor.size(0)), dtype=dtype, device=self.device)

            # img_B_tensor 的标签值应为 1
            labels_B =  torch.ones((img_A_tensor.size(0)), dtype=dtype, device=self.device)
            
            # 合并输入和标签
            inputs = torch.cat((img_B_tensor, img_A_tensor), dim=0)
            labels = torch.cat((labels_B, labels_A), dim=0)
            
            
            # inference
            # 前向传播
            outputs = self.model(inputs)

            # 计算损失函数
            loss = self.compute_loss(outputs, labels)
            
                # 计算预测值
            # 假设模型输出为 [batch_size, 1]，需要调整形状
            outputs = outputs.view(-1)
            preds = torch.sign(outputs)
            
                # 计算准确率
            correct = (preds == labels).sum().item()
            total = labels.size(0)

            # 更新指标
            metrics_total['loss'] += loss.item()
            metrics_total['correct'] += correct
            metrics_total['total'] += total
            
            # 将张量转换为列表,并格式化
            outputs_value = outputs.tolist()
            labels_value = labels.tolist()
            formatted_outputs_value =[float(f'{num:.2f}') for num in outputs_value]
            formatted_accuracy_value = float(f'{(metrics_total['correct']/metrics_total['total']):.4f}')

            # 更新进度条信息
            loop_valid.set_description(f"Valid Epoch [{self.epoch_done}/{self.config.opt.epoch_end}]")
            # loop_valid.set_postfix(loss=loss.item(),accuracy=accuracy)
            loop_valid.set_postfix(loss=loss.item(),correct=metrics_total['correct'],total=metrics_total['total'],accuracy=formatted_accuracy_value)
            

            # 这里可以设置是否启用web服务（Visdom）
            # self.logger.log(losses=train_indexs, images=train_images)
            
            
            # 合并损失
            # total_loss = self.loss_G + self.loss_D
            # total_loss.backward()
            
            # # 反向传播和优化
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            
        # 计算并打印当前 epoch 的平均损失和准确率
        avg_loss = metrics_total['loss'] / self.train_data_length
        avg_accuracy = metrics_total['correct'] / metrics_total['total']
        # print(f"Epoch [{epoch}/{self.config.opt.epoch_end}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        result_avg = {'epoch':self.epoch_done,'loss':avg_loss,'accuracy':avg_accuracy}
        self.save_dict_as_csv(result_avg,'log_valid.csv')
        
    def valid_loop_2(self):
        
         # Tensor type
        Tensor = self.Tensor
        prev_time = time.time()
        self.model.eval()
        
        metrics_total = self.initialize_metrics()
        loop_valid = tqdm(enumerate(self.dataloader_valid_2), total=self.valid_data_length_2, colour='yellow', ncols=self.config.cmd_bytes)
        for i, batch in loop_valid:
            # Model inputs
            img_A_tensor = batch["A"].type(Tensor) # To A
            img_B_tensor = batch["B"].type(Tensor) # From B 
            # img_B_tensor = torch.autograd.Variable(batch["B"].type(Tensor)) # From B 
            # COMPLETE THIS
            # img_A_tensor 的标签值应为-1,请创建相应的标签值
            # img_B_tensor 的标签值应为1,请创建相应的标签值
            # 计算损失函数的方法是self.compute_loss(),可以直接调用，补充完成后续的代码
            
            # 创建标签
            # img_A_tensor 的标签值应为 -1
            labels_A = torch.tensor((img_A_tensor.size(0)).fill_(-1),dtype=Tensor,device=self.device)
            device = img_A_tensor.device
            dtype = img_A_tensor.dtype
            labels_A = -torch.ones(img_A_tensor.size(0), dtype=dtype, device=device)


            # img_B_tensor 的标签值应为 1
            labels_B = Tensor(img_B_tensor.size(0)).fill_(1)
            
            # 合并输入和标签
            inputs = torch.cat((img_A_tensor, img_B_tensor), dim=0)
            labels = torch.cat((labels_A, labels_B), dim=0)
            
            
            # inference
            # 前向传播
            outputs = self.model(inputs)

            # 计算损失函数
            loss = self.compute_loss(outputs, labels)
            
                # 计算预测值
            # 假设模型输出为 [batch_size, 1]，需要调整形状
            outputs = outputs.view(-1)
            preds = torch.sign(outputs)
            
                # 计算准确率
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            accuracy = (correct * 1.0) / total

            # 更新指标
            metrics_total['loss'] += loss.item()
            metrics_total['correct'] += correct
            metrics_total['total'] += total

            # 更新进度条信息
            loop_valid.set_description(f"Epoch [{self.epoch_done}/{self.config.opt.epoch_end}]")
            # loop_valid.set_postfix(loss=loss.item(),accuracy=accuracy)
            loop_valid.set_postfix(loss=loss.item(),correct=metrics_total['correct'],total=metrics_total['total'],accuracy=accuracy)
            

            # 这里可以设置是否启用web服务（Visdom）
            # self.logger.log(losses=train_indexs, images=train_images)
            
            
            # 合并损失
            # total_loss = self.loss_G + self.loss_D
            # total_loss.backward()
            
            # # 反向传播和优化
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            
        # 计算并打印当前 epoch 的平均损失和准确率
        avg_loss = metrics_total['loss'] / self.train_data_length
        avg_accuracy = metrics_total['correct'] / metrics_total['total']
        # print(f"Epoch [{epoch}/{self.config.opt.epoch_end}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        result_avg = {'epoch':self.epoch_done,'loss':avg_loss,'accuracy':avg_accuracy}
        self.save_dict_as_csv(result_avg,'log_valid.csv')

        
    
    def compute_loss(self, y_pred,y_truth,x_input=None):
        # """计算损失函数，需在子类中实现"""
        # raise NotImplementedError("子类必须实现 compute_loss 方法")
        l1_loss_module = nn.L1Loss()
        l2_loss_module = nn.MSELoss()
        loss_l1 = l1_loss_module(y_pred,y_truth)
        loss_l2 = l2_loss_module(y_pred,y_truth)
        self.loss_lotal = 0.5*(loss_l1+loss_l2)

    def evaluate(self, epoch):
        """模型评估，可在子类中重写"""
        print(f"第 {epoch} 个周期的评估完成。")
        
    def print_model_info(self):
        """打印模型基本信息，包括名称和参数规模"""
        model_params = self.count_parameters(self.model)
        info = {
            'Model Name': self.model_name,
            'Model Info': {
                'Name': self.model.__class__.__name__,
                'Parameters': model_params
            },
            'H,W,C':{
                'H': self.H,
                'W': self.W,
                'C': self.C
            }
        }
        print(f'当前模型信息为：\n{info}')
        opt_dict = self.config.opt.__dict__
        print(f'当前模型超参数信息为：\n{opt_dict}')


    def save_model_info(self):
        """保存模型基本信息，包括名称和参数规模"""
        model_params = self.count_parameters(self.model)
        info = {
            'Vector Name': self.model_name,
            'Model Info': {
                'Name': self.model.__class__.__name__,
                'Parameters': model_params
            },
            'H,W,C':{
                'H': self.H,
                'W': self.W,
                'C': self.C
            }
        }
        os.makedirs(self.dir, exist_ok=True)
        
        info_path = os.path.join(self.dir, f'info({self.model_name}).txt')
        with open(info_path, 'w') as f:
            for key, value in info.items():
                f.write(f'{key}: {value}\n')
        print(f"GAN_base.save_model_info(): 模型信息已保存至 {info_path},")
        
        opt_path = os.path.join(self.dir, f'opt({self.model_name}).txt')
        opt_dict = self.config.opt.__dict__

        with open(opt_path, 'w') as f:
            for key, value in opt_dict.items():
                f.write(f'{key}: {value}\n')
        print(f"超参数信息已保存至 {opt_path},")
        

    def count_parameters(self, model):
        """计算模型的参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_checkpoint(self, epoch):
        """存档模型状态"""
        epoch = f'{epoch:03d}'
        M_pth = os.path.join(self.checkpoint_path, f'M_{self.model_name}_epoch_{epoch}.pth')
        # op_G_pth = os.path.join(checkpoint_path, f'op_G_{self.model_name}_epoch_{epoch}.pth')
        # op_D_pth = os.path.join(checkpoint_path, f'op_D_{self.model_name}_epoch_{epoch}.pth')

        torch.save(self.model.state_dict(),M_pth)
        # torch.save(self.optimizer_G.state_dict(),op_G_pth)
        # torch.save(self.optimizer_D.state_dict(),op_D_pth)
        print(f"模型存档已保存至 {self.checkpoint_path}")
        
    def get_random_tensor(self,H=256,W=256,C=3,B=1):
        # 创建一个形状为B=2, C=3, H=256, W=256的随机Tensor
        return torch.randn(B, C, H, W, device=self.device)
    
    def get_random_tensor_by_tuple(self,tu,B=1):
        return torch.randn(B, tu[2], tu[0], tu[1], device=self.device)

    def save_dict_as_csv(self,dict_data:dict,file_name_='default_name.csv'):
        """ 将字典以指定文件名保存为"self.dir路径下的csv文件，需要指定文件名"""
        
        filename = os.path.join(self.dir,file_name_)
        
        # 检查文件是否存在以决定是否需要写入表头
        file_exists = os.path.isfile(filename)
        
        # 打开文件，如果不存在则创建
        if not file_exists:
            with open(filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=dict_data.keys())
                
                # 如果文件是新创建的，写入表头
                if not file_exists:
                    writer.writeheader()
                
                # 写入数据
                writer.writerow(dict_data)
        else:
            # 读取CSV文件的列名
            with open(filename, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                existing_columns = next(reader)  # 读取第一行即列名

            # 检查列名与字典键是否完全匹配
            if sorted(existing_columns) == sorted(dict_data.keys()):
                # 如果匹配，添加新数据
                with open(filename, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=existing_columns)
                    writer.writerow(dict_data)
            else:
                print("列名不匹配，无法添加数据。")

    def visualize_samples(self, epoch, num_samples=64):
        """可视化采样结果"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            sample_path = os.path.join(self.sample_dir, f'{self.model_name}_epoch_{epoch}.png')
            save_image(fake_images, sample_path, nrow=8, normalize=True)
            print(f"生成的图片已保存至 {sample_path}")
        self.generator.train()
