import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2

from back.Base_Block_py import up_conv, Recurrent_block, RRCNN_block, Attention_block, UPAttention_block
from back.GAN_py import GAN_base
# 这个使用了密集的跳跃连接的多级unet
# 然后每个unet都有注意力机制 
# 后期通过每个不同的注意力的热力图 
# 来证明每个注意力侧重的是不同位置
# 判别器就是简单的Patchcnn判别器
        
# doubleunet +denseAttention最终版本
class DenseDoubleUnet_G(GAN_base):
    def __init__(self, img_ch=3, output_ch=3, t=2):
        super(DenseDoubleUnet_G, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # scale_factor：指定输出为输入的2倍数；
        self.RRCNN1_1 = RRCNN_block(ch_in=3, ch_out=64, t=t)
        self.RRCNN1_2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN1_3= RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN1_4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN1_5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.RRCNN2_1 = RRCNN_block(ch_in=3, ch_out=64, t=t)
        self.RRCNN2_2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN2_3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN2_4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN2_5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        # --------------------------------------------------------------------------

        self.Up1_5 = up_conv(ch_in=1024, ch_out=512)
        self.Att1_5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN1_5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up1_4 = up_conv(ch_in=512, ch_out=256)
        self.Att1_4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.UPAtt1_4_1=UPAttention_block(F_g=256, F_l=256, F_int=128,x_c=512,k=2)
        self.Up_RRCNN1_4 = RRCNN_block(ch_in=256*3, ch_out=256, t=t)

        self.Up1_3 = up_conv(ch_in=256, ch_out=128)
        self.Att1_3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.UPAtt1_3_1 = UPAttention_block(F_g=128, F_l=128, F_int=64, x_c=512, k=4)
        self.UPAtt1_3_2 = UPAttention_block(F_g=128, F_l=128, F_int=64, x_c=256, k=2)
        self.Up_RRCNN1_3 = RRCNN_block(ch_in=128*4, ch_out=128, t=t)

        self.Up1_2 = up_conv(ch_in=128, ch_out=64)
        self.Att1_2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.UPAtt1_2_1 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=512, k=8)
        self.UPAtt1_2_2 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=256, k=4)
        self.UPAtt1_2_3 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=128, k=2)
        self.Up_RRCNN1_2 = RRCNN_block(ch_in=64*5, ch_out=64, t=t)

        self.Conv1_1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        # --------------------------------------------------------------------------------------

        self.Up2_5 = up_conv(ch_in=1024, ch_out=512)
        self.Att2_5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Att3_5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN2_5 = RRCNN_block(ch_in=512*3, ch_out=512, t=t)

        self.Up2_4 = up_conv(ch_in=512, ch_out=256)
        self.Att2_4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att3_4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.UPAtt2_4_1 = UPAttention_block(F_g=256, F_l=256, F_int=128, x_c=512, k=2)
        self.UPAtt3_4_1 = UPAttention_block(F_g=256, F_l=256, F_int=128, x_c=512, k=2)
        self.Up_RRCNN2_4 = RRCNN_block(ch_in=256*5, ch_out=256, t=t)

        self.Up2_3 = up_conv(ch_in=256, ch_out=128)
        self.Att2_3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.UPAtt2_3_1 = UPAttention_block(F_g=128, F_l=128, F_int=64, x_c=512, k=4)
        self.UPAtt2_3_2 = UPAttention_block(F_g=128, F_l=128, F_int=64, x_c=256, k=2)
        self.UPAtt3_3_1 = UPAttention_block(F_g=128, F_l=128, F_int=64, x_c=512, k=4)
        self.UPAtt3_3_2 = UPAttention_block(F_g=128, F_l=128, F_int=64, x_c=256, k=2)
        self.Up_RRCNN2_3 = RRCNN_block(ch_in=128*7, ch_out=128, t=t)

        self.Up2_2 = up_conv(ch_in=128, ch_out=64)
        self.Att2_2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att3_2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.UPAtt2_2_1 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=512, k=8)
        self.UPAtt2_2_2 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=256, k=4)
        self.UPAtt2_2_3 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=128, k=2)
        self.UPAtt3_2_1 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=512, k=8)
        self.UPAtt3_2_2 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=256, k=4)
        self.UPAtt3_2_3 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=128, k=2)
        self.Up_RRCNN2_2 = RRCNN_block(ch_in=64*9, ch_out=64, t=t)

        self.Conv2_1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1_1(x)
        #3 256 256-64 256 256
        x2 = self.Maxpool(x1)

        #64 256 256- 64 128 128
        x2 = self.RRCNN1_2(x2)
        #64 128 128-128 128 128
        x3 = self.Maxpool(x2)
        #128 128 128-128 64 64
        x3 = self.RRCNN1_3(x3)
        #128 64 64-256 64 64
        x4 = self.Maxpool(x3)
        #256 64 64-256 32 32
        x4 = self.RRCNN1_4(x4)
        #256 32 32-512 32 32
        x5 = self.Maxpool(x4)
        #512 32 32-512 16 16
        x5 = self.RRCNN1_5(x5)
        #512 16 16-1024 16 16
        # ------------------------------------
        # decoding + concat path
        d1_5 = self.Up1_5(x5)
        #1024 16 16-512 32 32
        x1_4 = self.Att1_5(g=d1_5, x=x4)
        #512 32 32
        d1_5 = torch.cat((x1_4, d1_5), dim=1)
        #1024 32 32
        d1_5 = self.Up_RRCNN1_5(d1_5)
        #1024 32 32-512 32 32


        d1_4 = self.Up1_4(d1_5)
        #512 32 32-256 64 64
        x1_3 = self.Att1_4(g=d1_4, x=x3)
        up1_4=self.UPAtt1_4_1(g=d1_4, x=x4)
        #256 64 64
        d1_4 = torch.cat((up1_4,x1_3, d1_4), dim=1)
        #512 64 64
        d1_4 = self.Up_RRCNN1_4(d1_4)
        #256 64 64


        d1_3 = self.Up1_3(d1_4)
        #128 128 128
        x1_2 = self.Att1_3(g=d1_3, x=x2)
        up1_4=self.UPAtt1_3_1(g=d1_3, x=x4)
        up1_3 = self.UPAtt1_3_2(g=d1_3, x=x3)
        #128 128 128
        d1_3 = torch.cat((up1_3,up1_4,x1_2, d1_3), dim=1)
        #256 128 128
        d1_3 = self.Up_RRCNN1_3(d1_3)
        # 256 128 128-128 128 128


        d1_2 = self.Up1_2(d1_3)
        #64 256 256
        x1_1 = self.Att1_2(g=d1_2, x=x1)
        up1_4 = self.UPAtt1_2_1(g=d1_2, x=x4)
        up1_3 = self.UPAtt1_2_2(g=d1_2, x=x3)
        up1_2=self.UPAtt1_2_3(g=d1_2, x=x2)
        #64 256 256
        d1_2 = torch.cat(( up1_2, up1_3, up1_4,x1_1, d1_2), dim=1)
        #64 256 256-128 256 256
        d1_2 = self.Up_RRCNN1_2(d1_2)
        #128 256 256- 64 256 256

        mid_x = self.Conv1_1(d1_2)
        #64 256 256 -3 256 256
        # ----------------------------------------------

        mx1 = self.RRCNN2_1(mid_x)
        # 3 256 256-64 256 256
        mx2 = self.Maxpool(mx1)
        # 64 256 256- 64 128 128
        mx2 = self.RRCNN2_2(mx2)
        # 64 128 128-128 128 128
        mx3 = self.Maxpool(mx2)
        # 128 128 128-128 64 64
        mx3 = self.RRCNN2_3(mx3)
        # 128 64 64-256 64 64
        mx4 = self.Maxpool(mx3)
        # 256 64 64-256 32 32
        mx4 = self.RRCNN2_4(mx4)
        # 256 32 32-512 32 32
        mx5 = self.Maxpool(mx4)
        # 512 32 32-512 16 16
        mx5 = self.RRCNN2_5(mx5)
        # 512 16 16-1024 16 16
        # --------------------------------------

        # decoding + concat path
        d2_5 = self.Up2_5(mx5)
        # 1024 16 16-512 32 32
        x2_4 = self.Att2_5(g=d2_5, x=mx4)
        # 512 32 32
        x3_4 = self.Att3_5(g=d2_5, x=x4)
        # 512 32 32
        d2_5 = torch.cat((x3_4,x2_4, d2_5), dim=1)
        # d2_5 = torch.cat((x3_4, x3_4, d2_5), dim=1)
        # 1536 32 32
        d2_5 = self.Up_RRCNN2_5(d2_5)
        # 1536 32 32-512 32 32

        d2_4 = self.Up2_4(d2_5)
        # 512 32 32-256 64 64
        x2_3 = self.Att2_4(g=d2_4, x=mx3)
        # 256 64 64
        x3_3 = self.Att3_4(g=d2_4, x=x3)
        up2_4=self.UPAtt2_4_1(g=d2_4, x=mx4)
        up3_4 =self.UPAtt3_4_1(g=d2_4, x=x4)

        # 256 64 64
        d2_4 = torch.cat((up3_4,up2_4,x3_3,x2_3, d2_4), dim=1)
        # d2_4 = torch.cat((up3_4, up3_4, x3_3, x3_3, d2_4), dim=1)
        # 768 64 64
        d2_4 = self.Up_RRCNN2_4(d2_4)
        # 256 64 64


        d2_3 = self.Up2_3(d2_4)
        # 128 128 128
        x2_2 = self.Att2_3(g=d2_3, x=mx2)
        # 128 128 128
        x3_2 = self.Att3_3(g=d2_3, x=x2)
        # 128 128 128
        up2_4=self.UPAtt2_3_1(g=d2_3, x=mx4)
        up2_3 =self.UPAtt2_3_2(g=d2_3, x=mx3)
        up3_4 =self.UPAtt3_3_1(g=d2_3, x=x4)
        up3_3=self.UPAtt3_3_2(g=d2_3, x=x3)

        d2_3 = torch.cat((up3_3,up3_4,up2_3,up2_4,x3_2,x2_2, d2_3), dim=1)
        # d2_3 = torch.cat((up3_3, up3_4, up3_3, up3_4, x3_2, x3_2, d2_3), dim=1)
        # 384 128 128
        d2_3 = self.Up_RRCNN2_3(d2_3)
        # 384 128 128-128 128 128

        d2_2 = self.Up2_2(d2_3)
        # 64 256 256
        x2_1 = self.Att2_2(g=d2_2, x=mx1)
        # 64 256 256
        x3_1 = self.Att3_2(g=d2_2, x=x1)
        # 64 256 256
        up2_4=self.UPAtt2_2_1(g=d2_2, x=mx4)
        up2_3 =self.UPAtt2_2_2(g=d2_2, x=mx3)
        up2_2 =self.UPAtt2_2_3(g=d2_2, x=mx2)
        up3_4 =self.UPAtt3_2_1(g=d2_2, x=x4)
        up3_3 =self.UPAtt3_2_2(g=d2_2, x=x3)
        up3_2 =self.UPAtt3_2_3(g=d2_2, x=x2)

        d2_2 = torch.cat((up3_2,up3_3,up3_4,up2_2,up2_3,up2_4,x3_1,x2_1, d2_2), dim=1)
        # d2_2 = torch.cat((up3_2, up3_3, up3_4, up3_2, up3_3, up3_4, x3_1, x3_1, d2_2), dim=1)
        # 64 256 256-192 256 256
        d2_2 = self.Up_RRCNN2_2(d2_2)
        # 192 256 256- 64 256 256
        x = self.Conv2_1(d2_2)
        return x,mid_x,x

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
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
            # nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# d=Discriminator()
# img = torch.randn(1, 3, 256, 256)
# preds = d(img,img)
# print(preds.shape)



# x=torch.ones(1,3,256,256)
# net=DenseDoubleUnet()
# xx,yy=net(x)
# print(xx.shape)
# ---------------------------------------普通判别器----------------------------------------------------------
class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3*256*256, 512),                   ## 输入特征数为784，输出为512
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(512, 256),                        ## 输入特征数为512，输出为256
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(256, 1),                          ## 输入特征数为256，输出为1
            nn.Sigmoid(),                               ## sigmoid是一个激活函数，二分类问题中可将实数映射到[0, 1],作为概率值, 多分类用softmax函数
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)            ## 鉴别器输入是一个被view展开的(784)的一维图像:(64, 784)
        validity = self.model(img_flat)                 ## 通过鉴别器网络
        return validity                                 ## 鉴别器返回的是一个[0, 1]间的概率

# d=Discriminator2()
# img = torch.randn(1, 3, 256, 256)
# preds = d(img)
# print(preds.shape)