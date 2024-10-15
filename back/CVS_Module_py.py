import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

class CVS_Module(nn.Module):
    """ 集成该类的所有子类都应在定义变量后进行显式的init操作。用于显示的将当前全局配置信息传给本类。 """
    def __init__(self, model_name=None):
        super(CVS_Module, self).__init__()
        # 模型名称，默认为类名
        if model_name is None:
            self.model_name = self.__class__.__name__
        else:
            self.model_name = model_name
        self.config = None

    
    def init(self,config):
        self.config = config
        self.dir = os.path.join(f"exp_record/{self.config.time_stamp}")
        self.checkpoint_path = os.path.join(self.dir,'checkpoints')
        self.sample_path = os.path.join(self.dir,'samples')
        self.H = config.opt.H
        self.W = config.opt.W
        self.C = config.opt.C
        self.patch = (1, self.H // 2 ** 4, self.W // 2 ** 4)
        self.epoch_done = 0
        
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        
          # Configure dataloaders
        self.transforms_ = [
            transforms.Resize((self.H, self.W), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        
        
    def weights_init_normal(self,m):
        classname = m.__class__.__name__
        # type_name = type(m)
        # save_log_csv({"pos":"all","type_name":type_name,"classname":classname})
        if classname.find("Conv") != -1:
            # save_log_csv({"pos":"Conv","type_name":type_name,"classname":classname})
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # save_log_csv({"pos":"BatchNorm2d","type_name":type_name,"classname":classname})
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
            
    def display_model_info(self):
        """保存模型基本信息，包括名称和参数规模"""
        m_params = self.count_parameters(self)
        # disc_params = self.count_parameters(self.discriminator)
        info = {
            'Model Name': self.model_name,
            'Information': {
                'Name': self.__class__.__name__,
                'Parameters': m_params
            }
        }
        print(info)
        pass
        
        
        
    # def save_model_info(self):
    #     """保存模型基本信息，包括名称和参数规模"""
    #     model_info = {'Model Name': self.model_name}

    #     # 计算模型参数规模
    #     total_params = self.count_parameters()
    #     model_info['Total Parameters'] = total_params

    #     # 保存信息到文本文件
    #     info_path = os.path.join(self.checkpoint_path, f'info_{self.model_name}.txt')
    #     with open(info_path, 'w') as f:
    #         for key, value in model_info.items():
    #             f.write(f'{key}: {value}\n')
    #     print(f"模型信息已保存至 {info_path}")

    # def count_parameters(self):
    #     """计算模型的参数数量"""
    #     return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # def save_checkpoint(self, epoch):
    #     """存档模型状态"""
    #     epoch = f"{epoch:03d}"
    #     checkpoint_path = os.path.join(self.checkpoint_path, f'{self.model_name}_epoch_{epoch}.pth')

    #     torch.save(self.state_dict(), checkpoint_path)
    #     print(f"模型存档已保存至 {checkpoint_path}")

    # def train_model(self, dataloader, num_epochs, optimizer):
    #     """模型训练"""
    #     self.train()
    #     for epoch in range(1, num_epochs + 1):
    #         for i, data in enumerate(dataloader):
    #             optimizer.zero_grad()
    #             loss = self.compute_loss(data)
    #             loss.backward()
    #             optimizer.step()
    #         self.save_checkpoint(epoch)
    #         self.evaluate(epoch)
    #         self.visualize_samples(epoch)
    #         print(f"第 {epoch} 个周期完成。")

    # def compute_loss(self, data):
    #     """计算损失函数，需在子类中实现"""
    #     # raise NotImplementedError("子类必须实现 compute_loss 方法")
    #     print(f"compute_loss 方法")

    # def evaluate(self, epoch):
    #     """模型评估，可在子类中重写"""
    #     print(f"第 {epoch} 个周期的评估完成。")

    # def visualize_samples(self, epoch):
    #     """可视化采样结果，可在子类中实现"""
    #     pass
