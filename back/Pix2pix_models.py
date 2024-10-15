import torch
import torch.nn as nn
import torch.nn.functional as F
from back.GAN_py import GAN_base


class Pix2pix(GAN_base):
    def __init__(self, in_channels=3, out_channels=3):
        super(Pix2pix, self).__init__()
        self.generator = Pix2Pix_Generator(in_channels, out_channels)
        self.discriminator = Discriminator(in_channels)
        # self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        # self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.loss_G = 0
        self.loss_D = 0
        self.loss1 = nn.L1Loss() #L1Loss
        self.loss2 = nn.MSELoss() #L2loSS
        
    def compute_loss(self, pred_y, truth_y, x):
        # Discriminator loss
        real_output = self.discriminator(x, truth_y)
        fake_output = self.discriminator(x, pred_y.detach())
        d_loss_real = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        d_loss_fake = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        self.loss_D = d_loss
        g_loss = self.loss1(pred_y,truth_y)
        
        
        self.loss_G = g_loss
        

    # def compute_loss(self, pred_y, truth_y, x):
    #     """计算损失函数，需在子类中实现"""
    #     # Discriminator loss
    #     real_output = self.discriminator(x, truth_y)
    #     fake_output = self.discriminator(x, pred_y.detach())
    #     d_loss_real = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
    #     d_loss_fake = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
    #     d_loss = (d_loss_real + d_loss_fake) / 2

    #     # Generator loss
    #     g_loss = F.binary_cross_entropy(self.discriminator(x, pred_y), torch.ones_like(fake_output))
        
    #     return g_loss, d_loss
    

class UNetBlock(nn.Module):
    """ U-Net 编码器和解码器块 """
    def __init__(self, in_channels, out_channels, down=True, relu=True, batch_norm=True, dropout=0.0):
        super(UNetBlock, self).__init__()
        self.down = down
        layers = []
        if relu:
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not batch_norm))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not batch_norm))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input=None):
        x = self.model(x)
        if self.down:
            return x
        else:
            x = torch.cat([x, skip_input], 1)
            return x

class Pix2Pix_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Pix2Pix_Generator, self).__init__()
        # U-Net 下采样部分
        self.down1 = UNetBlock(in_channels, 64, down=True, relu=False, batch_norm=False)
        self.down2 = UNetBlock(64, 128, down=True)
        self.down3 = UNetBlock(128, 256, down=True)
        self.down4 = UNetBlock(256, 512, down=True)
        self.down5 = UNetBlock(512, 512, down=True)
        self.down6 = UNetBlock(512, 512, down=True)
        self.down7 = UNetBlock(512, 512, down=True)
        self.down8 = UNetBlock(512, 512, down=True, batch_norm=False)
        
        # U-Net 上采样部分
        self.up1 = UNetBlock(512, 512, down=False, dropout=0.5)
        self.up2 = UNetBlock(1024, 512, down=False, dropout=0.5)
        self.up3 = UNetBlock(1024, 512, down=False, dropout=0.5)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()  # 输出使用 Tanh 激活函数
        )

    def forward(self, x):
        # U-Net 向前传递
        input_x = x.clone()
        d1 = self.down1(input_x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        output = self.final(u7)
        
        return output,output,output
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            #layers.append(nn.Dropout(0.5))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)