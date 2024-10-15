import datetime
import argparse
import sys
import os
import torch

class Config:
    def __init__(self):
        self.opt = self.parse_args()
        self.time_stamp = self.opt.time_stamp
        self.path_exp_record = os.path.join(f"exp_record/{self.time_stamp}")
        self.cuda = torch.cuda.is_available()
        self.device_ids = list(range(torch.cuda.device_count()))  # 获取所有可用GPU的索引
        self.cmd_bytes = 168
        pass
    
    def show_cuda_status(self):
        print(f"cuda:{self.cuda}\ndevice_ids：{self.device_ids}")
        
    def adapt_cuda(self,m):
        # Assuming model is your neural network model
        if self.cuda:
            m = m.cuda()  # Move model to GPU
            if len(self.device_ids) > 1:
                # Use DataParallel to utilize multiple GPUs
                print("Use DataParallel to utilize multiple GPUs")
                m = torch.nn.DataParallel(m, device_ids=self.device_ids)
        else:
            # No GPU available, keep the model on CPU
            print("No GPU available, keep the model on CPU")
            pass  # Model is already on CPU by default
        
        return m
    
    def now(self):
        now = datetime.datetime.now()
        now_str = now.strftime('%Y_%m_%d_%H_%M_%S')
        return now_str
    
    def parse_args(self):
        current_datetime = datetime.datetime.now()
        auto_time_stamp = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        parser = argparse.ArgumentParser(description="parameters of training model.")
        parser.add_argument("--time_stamp", type=str, default=auto_time_stamp, help="时间戳")
        parser.add_argument("--pth_G", type=str, default=None, help="从指定存档开始G")
        parser.add_argument("--pth_D", type=str, default=None, help="从指定存档开始D")
        parser.add_argument("--epoch_start", type=int, default=0, help="起始训练轮数")
        parser.add_argument("--epoch_end", type=int, default=201, help="训练的总轮数")
        parser.add_argument("--dataset_name", type=str, default="dataset_AB", help="数据集名称")
        parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
        parser.add_argument("--num_workers", type=int, default=0, help="数据加载时的工作线程数,不支持在根缩进方法中运行。")
        parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
        parser.add_argument("--b1", type=float, default=0.5, help="Adam优化器的beta1参数")
        parser.add_argument("--b2", type=float, default=0.999, help="Adam优化器的beta2参数")
        parser.add_argument("--decay_epoch", type=int, default=100, help="学习率衰减起始轮数")
        parser.add_argument("--H", type=int, default=256, help="图片高度")
        parser.add_argument("--W", type=int, default=256, help="图片宽度")
        parser.add_argument("--C", type=int, default=3, help="图像通道数")
        
        parser.add_argument("--sample_interval_valid", type=int, default=1, help="抽样间隔多少个Epoch,1则为每个Epoch均采样。本数值*训练集长度")
        parser.add_argument("--sample_interval_train", type=int, default=1, help="训练时，抽样间隔,本数值*训练集长度")
        parser.add_argument("--checkpoint_interval", type=int, default=1, help="模型保存间隔")
        parser.add_argument("--save_ssim", type=float, default=0.00, help="模型评估值大于本阈值时，保存模型")
        parser.add_argument("--load_in_memory", type=int, default=1, help="数据集初始化时，读取全部数据到内存中，默认为1。")
        opt = parser.parse_args()
        return opt
    
    def parse_args_initial(self):
        current_datetime = datetime.datetime.now()
        auto_time_stamp = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        parser = argparse.ArgumentParser(description="parameters of training model.")
        parser.add_argument("--time_str", type=str, default=auto_time_stamp, help="指定时间")
        parser.add_argument("--pth", type=str, default=None, help="从指定存档开始")
        parser.add_argument("--checkpoint", type=int, default=0, help="从指定存档开始")
        parser.add_argument("--epoch", type=int, default=0, help="起始训练轮数")
        parser.add_argument("--n_epochs", type=int, default=201, help="训练的总轮数")
        parser.add_argument("--dataset_name", type=str, default="dataset", help="数据集名称")
        parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
        parser.add_argument("--num_workers", type=int, default=0, help="数据加载时的工作线程数")
        parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
        parser.add_argument("--b1", type=float, default=0.5, help="Adam优化器的beta1参数")
        parser.add_argument("--b2", type=float, default=0.999, help="Adam优化器的beta2参数")
        parser.add_argument("--decay_epoch", type=int, default=100, help="学习率衰减起始轮数")
        parser.add_argument("--img_height", type=int, default=256, help="图片高度")
        parser.add_argument("--img_width", type=int, default=256, help="图片宽度")
        parser.add_argument("--channels", type=int, default=3, help="图像通道数")
        parser.add_argument("--sample_interval", type=int, default=1, help="抽样间隔Epoch,本数值*训练集长度")
        parser.add_argument("--train_sample_interval", type=int, default=100, help="训练时，抽样间隔,本数值*训练集长度")
        parser.add_argument("--checkpoint_interval", type=int, default=1, help="模型保存间隔")
        parser.add_argument("--save_ssim", type=float, default=0.00, help="模型评估值大于本阈值时，保存模型")
        parser.add_argument("--aim", type=int, default=1, help="数据集初始化时，读取全部数据到内存中，默认为1。")
        # global opt
        opt = parser.parse_args()
        return opt
    
    def init_exp_record_GAN(self,pyfile,GAN):
        """ 初始化实验记录，保存当前运行的py文件名，生成器的信息，判别器的信息，运行的Nvidia显卡的信息 """
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpus_info = []
        for gpu in gpus:
            gpus_info.append(f"GPU ID:{gpu.id}, Name:{gpu.name}")
        
        # 获取当前文件的完整路径
        current_file_name = os.path.basename(pyfile)
        file_path = os.path.join(self.path_exp_record,f"record_{current_file_name}.txt")
        
        # 获取生成器和判别器的参数信息
        G_parameters,D_parameters = GAN.count_parameters()
        params_str = f"GAN name:{GAN.name}\nGenerator total parameters: {G_parameters}\nDiscriminator total parameters: {D_parameters}"
        
        print(f"Current file name:{current_file_name}\nexp_file_name:{file_path}\ngpus_info:{gpus_info}\nopt:{self.opt}\nParameter Info:{params_str}" )
        
        # 获取文件名部分
        with open(file_path, 'w') as file:
            file.write(f"exp_file_name:{file_path}\ngpus_info:{gpus_info}\nopt:{self.opt}\nParameter Info:{params_str}")  # 写入一些内容
            pass
    
    def function_empty(self):
        sys.stdout("This is a method of instance of Config. function name is function_empty().")
        pass