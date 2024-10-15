import torch
import torch.nn as nn
from back.GAN_py import GAN_base
from back.Base_Block_py import RRCNN_block, up_conv, Attention_block, UPAttention_block, conv_bn_relu

class UDCTS_Generator(GAN_base):
    """ 3阶段生成器，向上密集连接。
        注意：Level7的信息仅使用一个单连接，没有进行向上注意力连接。如果在生成器的三阶段中均加入带有L7信息的向上注意力连接，则模型参数两会增加约4亿。
        目前本模型的参数为：175,356,009
    """
    def __init__(self, img_ch=3, output_ch=3, t=2):
        super(UDCTS_Generator, self).__init__()

        #----- depthwise convolution 
        """
        [
        0: Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2), groups=64), 
        1: Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2), groups=128), 
        2: Conv2d(256, 256, kernel_size=(2, 2), stride=(2, 2), groups=256), 
        3: Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), groups=512), 
        4: Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2), groups=64), 
        5: Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2), groups=128), 
        6: Conv2d(256, 256, kernel_size=(2, 2), stride=(2, 2), groups=256), 
        7: Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), groups=512)
        ] 
        """
        # channels_ = [32,64,128,256,512,32,64,128,256,512,32,64,128,256,512]

        self.dwConv1_1 = nn.Conv2d(in_channels= 32,out_channels= 32,kernel_size=2,stride=2,padding=0,groups= 32)
        self.dwConv1_2 = nn.Conv2d(in_channels= 64,out_channels= 64,kernel_size=2,stride=2,padding=0,groups= 64)
        self.dwConv1_3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0,groups=128)
        self.dwConv1_4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0,groups=256)
        self.dwConv1_5 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=2,stride=2,padding=0,groups=512)
        
        self.dwConv2_1 = nn.Conv2d(in_channels= 32,out_channels= 32,kernel_size=2,stride=2,padding=0,groups= 32)
        self.dwConv2_2 = nn.Conv2d(in_channels= 64,out_channels= 64,kernel_size=2,stride=2,padding=0,groups= 64)
        self.dwConv2_3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0,groups=128)
        self.dwConv2_4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0,groups=256)
        self.dwConv2_5 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=2,stride=2,padding=0,groups=512)
        
        self.dwConv3_1 = nn.Conv2d(in_channels= 32,out_channels= 32,kernel_size=2,stride=2,padding=0,groups= 32)
        self.dwConv3_2 = nn.Conv2d(in_channels= 64,out_channels= 64,kernel_size=2,stride=2,padding=0,groups= 64)
        self.dwConv3_3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0,groups=128)
        self.dwConv3_4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0,groups=256)
        self.dwConv3_5 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=2,stride=2,padding=0,groups=512)
        #----- depthwise convolution end
        
        looptimes = 2
        # --- 20240530 start ---
        self.rrb1_1 = RRCNN_block(ch_in=1,ch_out=32,t=looptimes)
        """ ch_in=1 -> ch_out=3 """
        self.rrb1_2 = RRCNN_block(ch_in=32,ch_out=64,t=looptimes)
        self.rrb1_3 = RRCNN_block(ch_in=64,ch_out=128,t=looptimes)
        self.rrb1_4 = RRCNN_block(ch_in=128,ch_out=256,t=looptimes)
        self.rrb1_5 = RRCNN_block(ch_in=256,ch_out=512,t=looptimes)
        self.rrb1_6 = RRCNN_block(ch_in=512,ch_out=1024,t=looptimes)
        
        self.rrb2_1 = RRCNN_block(ch_in=1,ch_out=32,t=looptimes)
        self.rrb2_2 = RRCNN_block(ch_in=32,ch_out=64,t=looptimes)
        self.rrb2_3 = RRCNN_block(ch_in=64,ch_out=128,t=looptimes)
        self.rrb2_4 = RRCNN_block(ch_in=128,ch_out=256,t=looptimes)
        self.rrb2_5 = RRCNN_block(ch_in=256,ch_out=512,t=looptimes)
        self.rrb2_6 = RRCNN_block(ch_in=512,ch_out=1024,t=looptimes)
        
        self.rrb3_1 = RRCNN_block(ch_in=1,ch_out=32,t=looptimes)
        self.rrb3_2 = RRCNN_block(ch_in=32,ch_out=64,t=looptimes)
        self.rrb3_3 = RRCNN_block(ch_in=64,ch_out=128,t=looptimes)
        self.rrb3_4 = RRCNN_block(ch_in=128,ch_out=256,t=looptimes)
        self.rrb3_5 = RRCNN_block(ch_in=256,ch_out=512,t=looptimes)
        self.rrb3_6 = RRCNN_block(ch_in=512,ch_out=1024,t=looptimes)
        
        # --- 20240530  end ---
        
        # --- 20240604 start---
        # upconv series
        # ---
        self.upconv1_5 = up_conv(ch_in=1024,ch_out= 512)
        self.upconv1_4 = up_conv(ch_in= 512,ch_out= 256)
        self.upconv1_3 = up_conv(ch_in= 256,ch_out= 128)
        self.upconv1_2 = up_conv(ch_in= 128,ch_out=  64)
        self.upconv1_1 = up_conv(ch_in=  64,ch_out=  32)
        self.conv1_0 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.convbnrelu1_0 = conv_bn_relu(ch_in=32,ch_out=1,kernel=1,stride=1,padding=0,bias=True)
        
        self.upconv2_5 = up_conv(ch_in=1024,ch_out= 512)
        self.upconv2_4 = up_conv(ch_in= 512,ch_out= 256)
        self.upconv2_3 = up_conv(ch_in= 256,ch_out= 128)
        self.upconv2_2 = up_conv(ch_in= 128,ch_out=  64)
        self.upconv2_1 = up_conv(ch_in=  64,ch_out=  32)
        self.conv2_0 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.convbnrelu2_0 = conv_bn_relu(ch_in=32,ch_out=1,kernel=1,stride=1,padding=0,bias=True)
        
        self.upconv3_5 = up_conv(ch_in=1024,ch_out= 512)
        self.upconv3_4 = up_conv(ch_in= 512,ch_out= 256)
        self.upconv3_3 = up_conv(ch_in= 256,ch_out= 128)
        self.upconv3_2 = up_conv(ch_in= 128,ch_out=  64)
        self.upconv3_1 = up_conv(ch_in=  64,ch_out=  32)
        self.conv3_0 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.convbnrelu3_0 = conv_bn_relu(ch_in=32,ch_out=1,kernel=1,stride=1,padding=0,bias=True)
        
        self.last_conv = nn.Conv2d(3,3,kernel_size=1,stride=1,padding=0)
        
        # ---
        # --- 20240604  end ---
        
        # --------------------------------------------------------------------------  11 layer or block

        # --- 20240605 start---
        # 第一个编解码器使用的 ---------------------------------------
        self.atb1_5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.atb1_4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.atb1_3 = Attention_block(F_g=128, F_l=128, F_int= 64)
        self.atb1_2 = Attention_block(F_g= 64, F_l= 64, F_int= 32)
        self.atb1_1 = Attention_block(F_g= 32, F_l= 32, F_int= 16)
        self.atb1_0 = Attention_block(F_g=  1, F_l=  1, F_int=  1)
        
        # up_rrb 系列通常在cat操作后，用于降低通道数。
        self.up_rrb1_5 = RRCNN_block(ch_in= 512*2, ch_out=512, t=t)
        self.up_rrb1_4 = RRCNN_block(ch_in= 256*3, ch_out=256, t=t)
        self.up_rrb1_3 = RRCNN_block(ch_in= 128*4, ch_out=128, t=t)
        self.up_rrb1_2 = RRCNN_block(ch_in=  64*5, ch_out= 64, t=t)
        self.up_rrb1_1 = RRCNN_block(ch_in=  32*6, ch_out= 32, t=t)
        self.up_rrb1_0 = RRCNN_block(ch_in=  32, ch_out=  1, t=t)
        # self.up_rrb1_0 = RRCNN_block(ch_in=   1, ch_out=  1, t=t)
        
        self.up_atb1_5_5t4 = UPAttention_block(F_g=256, F_l=256, F_int=128, x_c= 512, k=2)
        self.up_atb1_4_4t3 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 256, k=2)
        self.up_atb1_4_5t3 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 512, k=4)
        self.up_atb1_3_3t2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 128, k=2)
        self.up_atb1_3_4t2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 256, k=4)
        self.up_atb1_3_5t2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 512, k=8)
        self.up_atb1_2_2t1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c=  64, k=2)
        self.up_atb1_2_3t1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 128, k=4)
        self.up_atb1_2_4t1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 256, k=8)
        self.up_atb1_2_5t1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 512, k=16)
        
        # 第二个编解码器使用的 ---------------------------------------
        self.atb2_5_1 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.atb2_5_2 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.atb2_4_1 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.atb2_4_2 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.atb2_3_1 = Attention_block(F_g=128, F_l=128, F_int= 64)
        self.atb2_3_2 = Attention_block(F_g=128, F_l=128, F_int= 64)
        self.atb2_2_1 = Attention_block(F_g= 64, F_l= 64, F_int= 32)
        self.atb2_2_2 = Attention_block(F_g= 64, F_l= 64, F_int= 32)
        self.atb2_1_1 = Attention_block(F_g= 32, F_l= 32, F_int= 16)
        self.atb2_1_2 = Attention_block(F_g= 32, F_l= 32, F_int= 16)
        self.atb2_0 = Attention_block(F_g=  1, F_l=  1, F_int=  1)
        
        # up_rrb 系列通常在cat操作后，用于降低通道数。
        self.up_rrb2_5 = RRCNN_block(ch_in= 512*3, ch_out=512, t=t)
        self.up_rrb2_4 = RRCNN_block(ch_in= 256*5, ch_out=256, t=t)
        self.up_rrb2_3 = RRCNN_block(ch_in= 128*7, ch_out=128, t=t)
        self.up_rrb2_2 = RRCNN_block(ch_in=  64*9, ch_out= 64, t=t)
        self.up_rrb2_1 = RRCNN_block(ch_in=  32*11, ch_out= 32, t=t)
        self.up_rrb2_0 = RRCNN_block(ch_in=  32, ch_out=  1, t=t)
        # self.up_rrb1_0 = RRCNN_block(ch_in=   1, ch_out=  1, t=t)
        
        self.up_atb2_5_5t4_1 = UPAttention_block(F_g=256, F_l=256, F_int=128, x_c= 512, k=2)
        self.up_atb2_5_5t4_2 = UPAttention_block(F_g=256, F_l=256, F_int=128, x_c= 512, k=2)
        
        self.up_atb2_4_4t3_1 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 256, k=2)
        self.up_atb2_4_5t3_1 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 512, k=4)
        self.up_atb2_4_4t3_2 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 256, k=2)
        self.up_atb2_4_5t3_2 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 512, k=4)
        
        self.up_atb2_3_3t2_1 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 128, k=2)
        self.up_atb2_3_4t2_1 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 256, k=4)
        self.up_atb2_3_5t2_1 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 512, k=8)
        self.up_atb2_3_3t2_2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 128, k=2)
        self.up_atb2_3_4t2_2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 256, k=4)
        self.up_atb2_3_5t2_2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 512, k=8)
        
        self.up_atb2_2_2t1_1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c=  64, k=2)
        self.up_atb2_2_3t1_1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 128, k=4)
        self.up_atb2_2_4t1_1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 256, k=8)
        self.up_atb2_2_5t1_1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 512, k=16)
        self.up_atb2_2_2t1_2 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c=  64, k=2)
        self.up_atb2_2_3t1_2 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 128, k=4)
        self.up_atb2_2_4t1_2 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 256, k=8)
        self.up_atb2_2_5t1_2 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 512, k=16)
        
        # 第三个编解码器使用的 ---------------------------------------
        self.atb3_5_1 = Attention_block(F_g=512, F_l=512, F_int=256) #  x用
        self.atb3_5_2 = Attention_block(F_g=512, F_l=512, F_int=256) # mx用
        self.atb3_5_3 = Attention_block(F_g=512, F_l=512, F_int=256) # nx用 下同
        
        self.atb3_4_1 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.atb3_4_2 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.atb3_4_3 = Attention_block(F_g=256, F_l=256, F_int=128)
        
        self.atb3_3_1 = Attention_block(F_g=128, F_l=128, F_int= 64)
        self.atb3_3_2 = Attention_block(F_g=128, F_l=128, F_int= 64)
        self.atb3_3_3 = Attention_block(F_g=128, F_l=128, F_int= 64)
        
        self.atb3_2_1 = Attention_block(F_g= 64, F_l= 64, F_int= 32)
        self.atb3_2_2 = Attention_block(F_g= 64, F_l= 64, F_int= 32)
        self.atb3_2_3 = Attention_block(F_g= 64, F_l= 64, F_int= 32)
        
        self.atb3_1_1 = Attention_block(F_g= 32, F_l= 32, F_int= 16)
        self.atb3_1_2 = Attention_block(F_g= 32, F_l= 32, F_int= 16)
        self.atb3_1_3 = Attention_block(F_g= 32, F_l= 32, F_int= 16)
                
        # up_rrb 系列通常在cat操作后，用于降低通道数。
        self.up_rrb3_5 = RRCNN_block(ch_in= 4*512, ch_out=512, t=t)
        self.up_rrb3_4 = RRCNN_block(ch_in= 7*256, ch_out=256, t=t)
        self.up_rrb3_3 = RRCNN_block(ch_in=10*128, ch_out=128, t=t)
        self.up_rrb3_2 = RRCNN_block(ch_in= 13*64, ch_out= 64, t=t)
        self.up_rrb3_1 = RRCNN_block(ch_in= 16*32, ch_out= 32, t=t)
        self.up_rrb3_0 = RRCNN_block(ch_in=    32, ch_out=  1, t=t)
        
        self.up_atb3_5_5t4_1 = UPAttention_block(F_g=256, F_l=256, F_int=128, x_c= 512, k=2) #  x用
        self.up_atb3_5_5t4_2 = UPAttention_block(F_g=256, F_l=256, F_int=128, x_c= 512, k=2) # mx用
        self.up_atb3_5_5t4_3 = UPAttention_block(F_g=256, F_l=256, F_int=128, x_c= 512, k=2) # nx用
        
        self.up_atb3_4_4t3_1 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 256, k=2)
        self.up_atb3_4_5t3_1 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 512, k=4)
        self.up_atb3_4_4t3_2 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 256, k=2)
        self.up_atb3_4_5t3_2 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 512, k=4)
        self.up_atb3_4_4t3_3 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 256, k=2)
        self.up_atb3_4_5t3_3 = UPAttention_block(F_g=128, F_l=128, F_int= 64, x_c= 512, k=4)
        
        self.up_atb3_3_3t2_1 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 128, k=2)
        self.up_atb3_3_4t2_1 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 256, k=4)
        self.up_atb3_3_5t2_1 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 512, k=8)
        self.up_atb3_3_3t2_2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 128, k=2)
        self.up_atb3_3_4t2_2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 256, k=4)
        self.up_atb3_3_5t2_2 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 512, k=8)
        self.up_atb3_3_3t2_3 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 128, k=2)
        self.up_atb3_3_4t2_3 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 256, k=4)
        self.up_atb3_3_5t2_3 = UPAttention_block(F_g= 64, F_l= 64, F_int= 32, x_c= 512, k=8)
        
        self.up_atb3_2_2t1_1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c=  64, k=2)
        self.up_atb3_2_3t1_1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 128, k=4)
        self.up_atb3_2_4t1_1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 256, k=8)
        self.up_atb3_2_5t1_1 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 512, k=16)
        self.up_atb3_2_2t1_2 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c=  64, k=2)
        self.up_atb3_2_3t1_2 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 128, k=4)
        self.up_atb3_2_4t1_2 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 256, k=8)
        self.up_atb3_2_5t1_2 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 512, k=16)
        self.up_atb3_2_2t1_3 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c=  64, k=2)
        self.up_atb3_2_3t1_3 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 128, k=4)
        self.up_atb3_2_4t1_3 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 256, k=8)
        self.up_atb3_2_5t1_3 = UPAttention_block(F_g= 32, F_l= 32, F_int= 16, x_c= 512, k=16)

        # --- 20240605  end ---     

    def forward(self, x):
        # rgb: 1 256 256 = 65536
        # x1 : 3 256 256 = 196608
        # x2 : 64 128 128 = 1048576
        # x3 : 128 64 64 = 524288
        # x4 : 256 32 32 = 262144
        # x5 : 512 16 16 = 131072
        # x6 : 1024 8 8 = 65536
        road1 = x[:, 0:1, :, :]
        # road2 = x[:, 1:2, :, :]
        # road3 = x[:, 2:3, :, :]
        
        # encoding path
        # phase 1-encoding
        road_times = 0
        x1 = self.rrb1_1(road1) # 1 256 256 -> 3 256 256
        x2 = self.dwConv1_1(x1) # 3 256 256 -> 3 128 128
        x2 = self.rrb1_2(x2) # 3 128 128 -> 64 128 128
        x3 = self.dwConv1_2(x2) # 64 128 128 -> 64 64 64
        x3 = self.rrb1_3(x3) # 64 64 64 -> 128 64 64
        x4 = self.dwConv1_3(x3) # 128 64 64 -> 128 32 32
        x4 = self.rrb1_4(x4) # 128 32 32 -> 256 32 32
        x5 = self.dwConv1_4(x4) # 256 32 32 -> 256 16 16
        x5 = self.rrb1_5(x5) # 256 16 16 -> 512 16 16
        x6 = self.dwConv1_5(x5) # 512 16 16 -> 512 8 8
        x6 = self.rrb1_6(x6) # 512 8 8 -> 1024 8 8
        
        # phase 1-5
        data1_5 = self.upconv1_5(x6) # 1024 8 8 -> 512 16 16
        x5_1 = self.atb1_5(g=data1_5,x=x5) # 512 16 16
        tmp1_5 = torch.cat((x5_1,data1_5),dim=1) # 1024 16 16
        data1_5 = self.up_rrb1_5(tmp1_5) # 1024 16 16 -> 512 16 16
        # phase 1-4
        data1_4 = self.upconv1_4(data1_5) # 512 16 16 -> 256 32 32
        x4_1 = self.atb1_4(g=data1_4,x=x4) # 256 32 32
        updata1_4 = self.up_atb1_5_5t4(g=data1_4,x=x5) # up_atb 会改变数据形状，从x5级别改为x4级别
        tmp1_4 = torch.cat((updata1_4,x4_1,data1_4),dim=1) # 256*3 32 32
        data1_4 = self.up_rrb1_4(tmp1_4) # 256*3 32 32 -> 256 32 32
        # phase 1-3
        data1_3 = self.upconv1_3(data1_4) # 256 32 32 -> 128 64 64
        x3_1 = self.atb1_3(g=data1_3,x=x3) # 128 64 64
        updata1_3_1 = self.up_atb1_4_4t3(g=data1_3,x=x4) # 128 64 64
        updata1_3_2 = self.up_atb1_4_5t3(g=data1_3,x=x5) # 128 64 64
        tmp1_3 = torch.cat((updata1_3_1,updata1_3_2,x3_1,data1_3),dim=1) # 128*4 64 64
        data1_3 = self.up_rrb1_3(tmp1_3)
        # phase 1-2
        data1_2 = self.upconv1_2(data1_3) # 64 128 128
        x2_1 = self.atb1_2(g=data1_2,x=x2)
        updata1_2_1 = self.up_atb1_3_3t2(g=data1_2,x=x3)
        updata1_2_2 = self.up_atb1_3_4t2(g=data1_2,x=x4)
        updata1_2_3 = self.up_atb1_3_5t2(g=data1_2,x=x5)
        tmp1_2 = torch.cat((updata1_2_1,updata1_2_2,updata1_2_3,x2_1,data1_2),dim=1)
        data1_2 = self.up_rrb1_2(tmp1_2)
        # phase 1-1
        data1_1 = self.upconv1_1(data1_2) # 32 256 256
        x1_1 = self.atb1_1(g=data1_1,x=x1)
        updata1_1_1 = self.up_atb1_2_2t1(g=data1_1,x=x2)
        updata1_1_2 = self.up_atb1_2_3t1(g=data1_1,x=x3)
        updata1_1_3 = self.up_atb1_2_4t1(g=data1_1,x=x4)
        updata1_1_4 = self.up_atb1_2_5t1(g=data1_1,x=x5)
        tmp1_1 = torch.cat((updata1_1_1,updata1_1_2,updata1_1_3,updata1_1_4,x1_1,data1_1),dim=1)
        data1_1 = self.up_rrb1_1(tmp1_1) # 32 256 256
        # phase 1-0
        mid1_0 = self.conv1_0(data1_1)
        
        # encoding path
        # phase 2-encoding
        road_times = 1
        mx1 = self.rrb1_1(mid1_0) # 1 256 256 -> 3 256 256
        mx2 = self.dwConv2_1(mx1) # 3 256 256 -> 3 128 128
        mx2 = self.rrb1_2(mx2) # 3 128 128 -> 64 128 128
        mx3 = self.dwConv2_2(mx2) # 64 128 128 -> 64 64 64
        mx3 = self.rrb1_3(mx3) # 64 64 64 -> 128 64 64
        mx4 = self.dwConv2_3(mx3) # 128 64 64 -> 128 32 32
        mx4 = self.rrb1_4(mx4) # 128 32 32 -> 256 32 32
        mx5 = self.dwConv2_4(mx4) # 256 32 32 -> 256 16 16
        mx5 = self.rrb1_5(mx5) # 256 16 16 -> 512 16 16
        mx6 = self.dwConv2_5(mx5) # 512 16 16 -> 512 8 8
        mx6 = self.rrb1_6(mx6) # 512 8 8 -> 1024 8 8
        
        # phase 2-5
        data2_5 = self.upconv2_5(mx6) # 512 16 16
        mx5_1 = self.atb2_5_1(g=data2_5,x= x5)
        mx5_2 = self.atb2_5_2(g=data2_5,x=mx5)
        tmp2_5 = torch.cat((mx5_1,mx5_2,data2_5),dim=1)
        data2_5 = self.up_rrb2_5(tmp2_5)
        # phase 2-4
        data2_4 = self.upconv2_4(data2_5) # 256 32 32
        updata2_4_1 = self.up_atb2_5_5t4_1(g=data2_4,x=x5)
        updata2_4_2 = self.up_atb2_5_5t4_2(g=data2_4,x=mx5)
        mx4_1 = self.atb2_4_1(g=data2_4,x=x4)
        mx4_2 = self.atb2_4_2(g=data2_4,x=mx4)
        tmp2_4 = torch.cat((updata2_4_1,updata2_4_2,mx4_1,mx4_2,data2_4),dim=1)
        data2_4 = self.up_rrb2_4(tmp2_4)
        # phase 2-3
        data2_3 = self.upconv2_3(data2_4) # 128 64 64
        updata2_3_1 = self.up_atb2_4_4t3_1(g=data2_3,x=x4)
        updata2_3_2 = self.up_atb2_4_5t3_1(g=data2_3,x=x5)
        updata2_3_3 = self.up_atb2_4_4t3_2(g=data2_3,x=mx4)
        updata2_3_4 = self.up_atb2_4_5t3_2(g=data2_3,x=mx5)
        mx3_1 = self.atb2_3_1(g=data2_3,x=x3)
        mx3_2 = self.atb2_3_2(g=data2_3,x=mx3)
        tmp2_3 = torch.cat((updata2_3_1,updata2_3_2,updata2_3_3,updata2_3_4,mx3_1,mx3_2,data2_3),dim=1)
        data2_3 = self.up_rrb2_3(tmp2_3)
        # phase 2-2
        data2_2 = self.upconv2_2(data2_3) # 64 128 128
        updata2_2_1 = self.up_atb2_3_3t2_1(g=data2_2,x=x3)
        updata2_2_2 = self.up_atb2_3_4t2_1(g=data2_2,x=x4)
        updata2_2_3 = self.up_atb2_3_5t2_1(g=data2_2,x=x5)
        updata2_2_4 = self.up_atb2_3_3t2_2(g=data2_2,x=mx3)
        updata2_2_5 = self.up_atb2_3_4t2_2(g=data2_2,x=mx4)
        updata2_2_6 = self.up_atb2_3_5t2_2(g=data2_2,x=mx5)
        mx2_1 = self.atb2_2_1(g=data2_2,x=x2)
        mx2_2 = self.atb2_2_2(g=data2_2,x=mx2)
        tmp2_2 = torch.cat((updata2_2_1,updata2_2_2,updata2_2_3,updata2_2_4,updata2_2_5,updata2_2_6,mx2_1,mx2_2,data2_2),dim=1)
        data2_2 = self.up_rrb2_2(tmp2_2)
        #phase 2-1
        data2_1 = self.upconv2_1(data2_2) # 32 256 256
        updata2_1_1 = self.up_atb2_2_2t1_1(g=data2_1,x= x2)
        updata2_1_2 = self.up_atb2_2_3t1_1(g=data2_1,x= x3)
        updata2_1_3 = self.up_atb2_2_4t1_1(g=data2_1,x= x4)
        updata2_1_4 = self.up_atb2_2_5t1_1(g=data2_1,x= x5)
        updata2_1_5 = self.up_atb2_2_2t1_2(g=data2_1,x=mx2)
        updata2_1_6 = self.up_atb2_2_3t1_2(g=data2_1,x=mx3)
        updata2_1_7 = self.up_atb2_2_4t1_2(g=data2_1,x=mx4)
        updata2_1_8 = self.up_atb2_2_5t1_2(g=data2_1,x=mx5)
        mx2_1 = self.atb2_1_1(g=data2_1,x= x1)
        mx2_2 = self.atb2_1_1(g=data2_1,x=mx1)
        tmp2_1 = torch.cat((updata2_1_1,updata2_1_2,updata2_1_3,updata2_1_4,updata2_1_5,updata2_1_6,updata2_1_7,updata2_1_8,mx2_1,mx2_2,data2_1), dim=1)
        data2_1 = self.up_rrb2_1(tmp2_1)
        #phase 2-0
        mid2_0 = self.conv2_0(data2_1)
        
        # encoding path
        # phase 3-encoding
        nx1 = self.rrb3_1(mid2_0) # 1 256 256 -> 3 256 256
        nx2 = self.dwConv3_1(nx1) # 3 256 256 -> 3 128 128
        nx2 = self.rrb3_2(nx2) # 3 128 128 -> 64 128 128
        nx3 = self.dwConv3_2(nx2) # 64 128 128 -> 64 64 64
        nx3 = self.rrb3_3(nx3) # 64 64 64 -> 128 64 64
        nx4 = self.dwConv3_3(nx3) # 128 64 64 -> 128 32 32
        nx4 = self.rrb3_4(nx4) # 128 32 32 -> 256 32 32
        nx5 = self.dwConv3_4(nx4) # 256 32 32 -> 256 16 16
        nx5 = self.rrb3_5(nx5) # 256 16 16 -> 512 16 16
        nx6 = self.dwConv3_5(nx5) # 512 16 16 -> 512 8 8
        nx6 = self.rrb3_6(nx6) # 512 8 8 -> 1024 8 8
        
        # phase 3-5
        data3_5 = self.upconv3_5(nx6) # 512 16 16
        nx5_1 = self.atb3_5_1(g=data3_5,x= x5)
        nx5_2 = self.atb3_5_2(g=data3_5,x=mx5)
        nx5_3 = self.atb3_5_3(g=data3_5,x=nx5)
        tmp3_5 = torch.cat((nx5_1,nx5_2,nx5_3,data3_5),dim=1)
        data3_5 = self.up_rrb3_5(tmp3_5)
        # phase 3-4
        data3_4 = self.upconv3_4(data3_5) # 256 32 32
        updata3_4_1 = self.up_atb3_5_5t4_1(g=data3_4,x= x5)
        updata3_4_2 = self.up_atb3_5_5t4_2(g=data3_4,x=mx5)
        updata3_4_3 = self.up_atb3_5_5t4_3(g=data3_4,x=nx5)
        nx4_1 = self.atb3_4_1(g=data3_4,x= x4)
        nx4_2 = self.atb3_4_2(g=data3_4,x=mx4)
        nx4_3 = self.atb3_4_3(g=data3_4,x=nx4)
        tmp3_4 = torch.cat((updata3_4_1,updata3_4_2,updata3_4_3,nx4_1,nx4_2,nx4_3,data3_4),dim=1)
        data3_4 = self.up_rrb3_4(tmp3_4)
        # phase 3-3
        data3_3 = self.upconv3_3(data3_4) # 128 64 64
        updata3_3_1 = self.up_atb3_4_4t3_1(g=data3_3,x= x4)
        updata3_3_2 = self.up_atb3_4_5t3_1(g=data3_3,x= x5)
        updata3_3_3 = self.up_atb3_4_4t3_2(g=data3_3,x=mx4)
        updata3_3_4 = self.up_atb3_4_5t3_2(g=data3_3,x=mx5)
        updata3_3_5 = self.up_atb3_4_4t3_3(g=data3_3,x=nx4)
        updata3_3_6 = self.up_atb3_4_5t3_3(g=data3_3,x=nx5)
        nx3_1 = self.atb3_3_1(g=data3_3,x= x3)
        nx3_2 = self.atb3_3_1(g=data3_3,x=mx3)
        nx3_3 = self.atb3_3_1(g=data3_3,x=nx3)
        tmp3_3 = torch.cat((updata3_3_1,updata3_3_2,updata3_3_3,updata3_3_4,updata3_3_5,updata3_3_6,nx3_1,nx3_2,nx3_3,data3_3),dim=1)
        data3_3 = self.up_rrb3_3(tmp3_3)
        # phase 3-2
        data3_2 = self.upconv3_2(data3_3) # 64 128 128
        updata3_2_1 = self.up_atb3_3_3t2_1(g=data3_2,x= x3)
        updata3_2_2 = self.up_atb3_3_4t2_1(g=data3_2,x= x4)
        updata3_2_3 = self.up_atb3_3_5t2_1(g=data3_2,x= x5)
        updata3_2_4 = self.up_atb3_3_3t2_2(g=data3_2,x=mx3)
        updata3_2_5 = self.up_atb3_3_4t2_2(g=data3_2,x=mx4)
        updata3_2_6 = self.up_atb3_3_5t2_2(g=data3_2,x=mx5)
        updata3_2_7 = self.up_atb3_3_3t2_3(g=data3_2,x=nx3)
        updata3_2_8 = self.up_atb3_3_4t2_3(g=data3_2,x=nx4)
        updata3_2_9 = self.up_atb3_3_5t2_3(g=data3_2,x=nx5)
        nx2_1 = self.atb3_2_1(g=data3_2,x= x2)
        nx2_2 = self.atb3_2_2(g=data3_2,x=mx2)
        nx2_3 = self.atb3_2_3(g=data3_2,x=nx2)
        tmp3_2 = torch.cat((updata3_2_1,updata3_2_2,updata3_2_3,updata3_2_4,updata3_2_5,updata3_2_6,updata3_2_7,updata3_2_8,updata3_2_9,nx2_1,nx2_2,nx2_3,data3_2),dim=1)
        data3_2 = self.up_rrb3_2(tmp3_2)
        # phase 3-1
        data3_1 = self.upconv3_1(data3_2) # 32 256 256
        updata3_1_1 = self.up_atb3_2_2t1_1(g=data3_1,x= x2)
        updata3_1_2 = self.up_atb3_2_3t1_1(g=data3_1,x= x3)
        updata3_1_3 = self.up_atb3_2_4t1_1(g=data3_1,x= x4)
        updata3_1_4 = self.up_atb3_2_5t1_1(g=data3_1,x= x5)
        updata3_1_5 = self.up_atb3_2_2t1_2(g=data3_1,x=mx2)
        updata3_1_6 = self.up_atb3_2_3t1_2(g=data3_1,x=mx3)
        updata3_1_7 = self.up_atb3_2_4t1_2(g=data3_1,x=mx4)
        updata3_1_8 = self.up_atb3_2_5t1_2(g=data3_1,x=mx5)
        updata3_1_9 = self.up_atb3_2_2t1_3(g=data3_1,x=nx2)
        updata3_1_10 = self.up_atb3_2_3t1_3(g=data3_1,x=nx3)
        updata3_1_11 = self.up_atb3_2_4t1_3(g=data3_1,x=nx4)
        updata3_1_12 = self.up_atb3_2_5t1_3(g=data3_1,x=nx5)
        nx1_1 = self.atb3_1_1(g=data3_1,x= x1)
        nx1_2 = self.atb3_1_2(g=data3_1,x=mx1)
        nx1_3 = self.atb3_1_3(g=data3_1,x=nx1)
        tmp3_1 = torch.cat((updata3_1_1,updata3_1_2,updata3_1_3,updata3_1_4,updata3_1_5,updata3_1_6,updata3_1_7,updata3_1_8,updata3_1_9,updata3_1_10,updata3_1_11,updata3_1_12,nx1_1,nx1_2,nx1_3,data3_1),dim=1)
        data3_1 = self.up_rrb3_1(tmp3_1)
        # phase 3-0
        mid3_0 = self.conv3_0(data3_1)

        return mid3_0,mid2_0,mid1_0