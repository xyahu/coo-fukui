import os
import torch
import numpy as np
from torch import nn
from torchvision.utils import save_image
from back.CVS_Module_py import CVS_Module
import csv

class Base_Model_Vector(CVS_Module):
    def __init__(self,c,m):
        super(Base_Model_Vector, self).__init__()
        self.config = c
        self.model = m
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
