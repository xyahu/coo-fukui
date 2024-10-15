import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import csv
import os
import torchvision



def limit_zero2one(data_):
    data = data_.clone()
    data[data>=1] = 1
    data[data<=-1] = -1
    data = (data + 1) / 2
    return data

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

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

def save_tensor_as_image(images,idx,folder_name="train_process"):
    # train_process_images = {'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A,'fake_mid2':fake_mid2, 'fake_mid1':fake_mid1}
    time_str = "default"
    dir_path = f"samples/{time_str}/{folder_name}"
    os.makedirs(dir_path, exist_ok=True) 
    dir_path_combined = f"samples/{time_str}/{folder_name}/combined"
    os.makedirs(dir_path_combined, exist_ok=True)
    combined_image = None

    for image_name, tensor in images.items():
        # temp_dir_path = f"samples/{time_str}/train_process/{idx}/{image_name}"
        # os.makedirs(temp_dir_path, exist_ok=True)
        
        
         # Normalize images to a [0, 1] scale
        dn_image = de_norm(tensor.data)

        # Stack all images vertically
        if combined_image is None:
            combined_image = dn_image
        else:
            combined_image = torch.cat([combined_image, dn_image], dim=2)
        
    # Save the combined image
    save_path = os.path.join(dir_path_combined, f"combined_{idx:08d}.png")
    # Save all images in one column
    torchvision.utils.save_image(combined_image, save_path) 

class CSVLogger():
    def __init__(self, n_epochs, batches_epoch, csv_file="./default.csv"):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.csv_file = csv_file
        self.init_csv()

    def init_csv(self):
        # 在初始化阶段，无法确定所有的loss名称，因此暂时不在这里写入头部
        pass
    
    def seconds_2_timedelta(self,seconds):
        eta_timedelta = datetime.timedelta(seconds=seconds)
        # 提取天数、秒数
        days = eta_timedelta.days
        total_seconds = eta_timedelta.total_seconds()

        # 从总秒数中剔除完整的天数
        seconds_remaining = total_seconds - (days * 86400)

        # 计算小时数
        hours = seconds_remaining // 3600
        seconds_remaining %= 3600

        # 计算分钟数
        minutes = seconds_remaining // 60

        # 计算剩余秒数并格式化到小数点后两位
        seconds = seconds_remaining % 60
        formatted_seconds = "{:02d}".format(int(seconds))

        # 格式化字符串为 DD:HH:MM:SS.ff
        formatted_eta = f"{days:02d}:{int(hours):02d}:{int(minutes):02d}:{formatted_seconds}"
        return formatted_eta
        

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].mean().item()
            else:
                self.losses[loss_name] += losses[loss_name].mean().item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        eta_seconds = batches_left * self.mean_period / batches_done
        formatted_eta = self.seconds_2_timedelta(eta_seconds)
        sys.stdout.write(f'ETA: {formatted_eta} | {self.mean_period / batches_done:.2f} s/it.')
        # sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))
        
        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of Epoch
        if (self.batch % self.batches_epoch) == 0:
            # Save losses to CSV at the end of each epoch
            self.save_losses_to_csv()

            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    options = {
                        'xlabel': 'Epochs',
                        'ylabel': loss_name,
                        'title': loss_name,
                        'xlim': [0, self.n_epochs],  # Assuming self.n_epochs is the total number of epochs
                        'ylim': [0, 'auto']  # Here 'auto' lets Visdom automatically adjust the upper limit based on data
                    }
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), opts=options)
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

    def save_losses_to_csv(self):
        if self.epoch == 1:  # 第一次保存时创建头部
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                headers = ['epoch'] + list(self.losses.keys())
                writer.writerow(headers)
        
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            formatted_losses = [f"{loss / self.batch:.4f}" for loss in self.losses.values()]
            row = [self.epoch] + formatted_losses
            writer.writerow(row)