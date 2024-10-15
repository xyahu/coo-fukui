import torch
import torch.nn as nn
from back.GAN_py import GAN_base
from back.Base_Block_py import RRCNN_block, up_conv, Attention_block, UPAttention_block, conv_bn_relu

class L6_Alpha_G(GAN_base):
    """ 3阶段生成器，向上密集连接。
        注意：Level7的信息也使用了向上注意力密集连接。在生成器的三阶段中均加入带有L7信息的向上注意力连接，因此本模型参数相对UDCTS模型增加约3.9亿参数。
        目前本模型的参数为：569,637,411
    """
    def __init__(self, img_ch=3, output_ch=3, t=2):
        super(L6_Alpha_G, self).__init__()

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
        # C_l0 = 1
        C_l1 = 1
        C_l2 = 32
        C_l3 = 64
        C_l4 = 128
        C_l5 = 256
        C_l6 = 512
        C_l7 = 1024

        self.dwConv1_1 = nn.Conv2d(in_channels=C_l2, out_channels=C_l2, kernel_size=2,stride=2,padding=0,groups=C_l2)
        self.dwConv1_2 = nn.Conv2d(in_channels=C_l3, out_channels=C_l3, kernel_size=2,stride=2,padding=0,groups=C_l3)
        self.dwConv1_3 = nn.Conv2d(in_channels=C_l4, out_channels=C_l4, kernel_size=2,stride=2,padding=0,groups=C_l4)
        self.dwConv1_4 = nn.Conv2d(in_channels=C_l5, out_channels=C_l5, kernel_size=2,stride=2,padding=0,groups=C_l5)
        self.dwConv1_5 = nn.Conv2d(in_channels=C_l6, out_channels=C_l6, kernel_size=2,stride=2,padding=0,groups=C_l6)
        
        self.dwConv2_1 = nn.Conv2d(in_channels=C_l2, out_channels=C_l2, kernel_size=2,stride=2,padding=0,groups=C_l2)
        self.dwConv2_2 = nn.Conv2d(in_channels=C_l3, out_channels=C_l3, kernel_size=2,stride=2,padding=0,groups=C_l3)
        self.dwConv2_3 = nn.Conv2d(in_channels=C_l4, out_channels=C_l4, kernel_size=2,stride=2,padding=0,groups=C_l4)
        self.dwConv2_4 = nn.Conv2d(in_channels=C_l5, out_channels=C_l5, kernel_size=2,stride=2,padding=0,groups=C_l5)
        self.dwConv2_5 = nn.Conv2d(in_channels=C_l6, out_channels=C_l6, kernel_size=2,stride=2,padding=0,groups=C_l6)
        
        self.dwConv3_1 = nn.Conv2d(in_channels=C_l2,out_channels=C_l2,kernel_size=2,stride=2,padding=0,groups=C_l2)
        self.dwConv3_2 = nn.Conv2d(in_channels=C_l3,out_channels=C_l3,kernel_size=2,stride=2,padding=0,groups=C_l3)
        self.dwConv3_3 = nn.Conv2d(in_channels=C_l4,out_channels=C_l4,kernel_size=2,stride=2,padding=0,groups=C_l4)
        self.dwConv3_4 = nn.Conv2d(in_channels=C_l5,out_channels=C_l5,kernel_size=2,stride=2,padding=0,groups=C_l5)
        self.dwConv3_5 = nn.Conv2d(in_channels=C_l6,out_channels=C_l6,kernel_size=2,stride=2,padding=0,groups=C_l6)
        #----- depthwise convolution end
        looptimes = 2
        # --- 20240530 start ---
        """ RRCNN_BLOCK """
        self.rrb1_1 = RRCNN_block(ch_in=C_l1,ch_out=C_l2,t=looptimes)
        self.rrb1_2 = RRCNN_block(ch_in=C_l2,ch_out=C_l3,t=looptimes)
        self.rrb1_3 = RRCNN_block(ch_in=C_l3,ch_out=C_l4,t=looptimes)
        self.rrb1_4 = RRCNN_block(ch_in=C_l4,ch_out=C_l5,t=looptimes)
        self.rrb1_5 = RRCNN_block(ch_in=C_l5,ch_out=C_l6,t=looptimes)
        self.rrb1_6 = RRCNN_block(ch_in=C_l6,ch_out=C_l7,t=looptimes)
        
        self.rrb2_1 = RRCNN_block(ch_in=C_l1,ch_out=C_l2,t=looptimes)
        self.rrb2_2 = RRCNN_block(ch_in=C_l2,ch_out=C_l3,t=looptimes)
        self.rrb2_3 = RRCNN_block(ch_in=C_l3,ch_out=C_l4,t=looptimes)
        self.rrb2_4 = RRCNN_block(ch_in=C_l4,ch_out=C_l5,t=looptimes)
        self.rrb2_5 = RRCNN_block(ch_in=C_l5,ch_out=C_l6,t=looptimes)
        self.rrb2_6 = RRCNN_block(ch_in=C_l6,ch_out=C_l7,t=looptimes)
        
        self.rrb3_1 = RRCNN_block(ch_in=C_l1,ch_out=C_l2,t=looptimes)
        self.rrb3_2 = RRCNN_block(ch_in=C_l2,ch_out=C_l3,t=looptimes)
        self.rrb3_3 = RRCNN_block(ch_in=C_l3,ch_out=C_l4,t=looptimes)
        self.rrb3_4 = RRCNN_block(ch_in=C_l4,ch_out=C_l5,t=looptimes)
        self.rrb3_5 = RRCNN_block(ch_in=C_l5,ch_out=C_l6,t=looptimes)
        self.rrb3_6 = RRCNN_block(ch_in=C_l6,ch_out=C_l7,t=looptimes)
        
        # --- 20240530  end ---
        
        # --- 20240604 start---
        # upconv series
        # ---
        self.upconv1_5 = up_conv(ch_in= C_l7,ch_out= C_l6)
        self.upconv1_4 = up_conv(ch_in= C_l6,ch_out= C_l5)
        self.upconv1_3 = up_conv(ch_in= C_l5,ch_out= C_l4)
        self.upconv1_2 = up_conv(ch_in= C_l4,ch_out= C_l3)
        self.upconv1_1 = up_conv(ch_in= C_l3,ch_out= C_l2)
        self.conv1_0 = nn.Conv2d(C_l2, C_l1, kernel_size=1, stride=1, padding=0)
        self.convbnrelu1_0 = conv_bn_relu(ch_in=C_l2,ch_out=C_l1,kernel=1,stride=1,padding=0,bias=True)
        
        self.upconv2_5 = up_conv(ch_in=C_l7,ch_out= C_l6)
        self.upconv2_4 = up_conv(ch_in= C_l6,ch_out= C_l5)
        self.upconv2_3 = up_conv(ch_in= C_l5,ch_out= C_l4)
        self.upconv2_2 = up_conv(ch_in= C_l4,ch_out=  C_l3)
        self.upconv2_1 = up_conv(ch_in=  C_l3,ch_out=  C_l2)
        self.conv2_0 = nn.Conv2d(C_l2, C_l1, kernel_size=1, stride=1, padding=0)
        self.convbnrelu2_0 = conv_bn_relu(ch_in=C_l2,ch_out=1,kernel=1,stride=1,padding=0,bias=True)
        
        self.upconv3_5 = up_conv(ch_in= C_l7,ch_out= C_l6)
        self.upconv3_4 = up_conv(ch_in= C_l6,ch_out= C_l5)
        self.upconv3_3 = up_conv(ch_in= C_l5,ch_out= C_l4)
        self.upconv3_2 = up_conv(ch_in= C_l4,ch_out= C_l3)
        self.upconv3_1 = up_conv(ch_in= C_l3,ch_out= C_l2)
        self.conv3_0 = nn.Conv2d(C_l2, C_l1, kernel_size=1, stride=1, padding=0)
        self.convbnrelu3_0 = conv_bn_relu(ch_in=C_l2,ch_out=C_l1,kernel=1,stride=1,padding=0,bias=True)
        
        # self.last_conv = nn.Conv2d(3,3,kernel_size=1,stride=1,padding=0)
        
        # ---
        # --- 20240604  end ---
        
        # --------------------------------------------------------------------------  11 layer or block

        # --- 20240605 start---
        # 第一个编解码器使用的 ---------------------------------------
        self.atb1_5 = Attention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5)
        self.atb1_4 = Attention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4)
        self.atb1_3 = Attention_block(F_g=C_l4, F_l=C_l4, F_int=C_l3)
        self.atb1_2 = Attention_block(F_g=C_l3, F_l=C_l3, F_int=C_l2)
        self.atb1_1 = Attention_block(F_g=C_l2, F_l=C_l2, F_int=C_l1)
        # self.atb1_0 = Attention_block(F_g=  1, F_l=  1, F_int=  1)
        
        # up_rrb 系列通常在cat操作后，用于降低通道数。
        self.up_rrb1_5 = RRCNN_block(ch_in= C_l6*3, ch_out=C_l6, t=t)
        self.up_rrb1_4 = RRCNN_block(ch_in= C_l5*4, ch_out=C_l5, t=t)
        self.up_rrb1_3 = RRCNN_block(ch_in= C_l4*5, ch_out=C_l4, t=t)
        self.up_rrb1_2 = RRCNN_block(ch_in= C_l3*6, ch_out=C_l3, t=t)
        self.up_rrb1_1 = RRCNN_block(ch_in= C_l2*7, ch_out=C_l2, t=t)
        # self.up_rrb1_0 = RRCNN_block(ch_in= C_l2  , ch_out=C_l1, t=t)
        # self.up_rrb1_0 = RRCNN_block(ch_in=   1, ch_out=  1, t=t)
        
        # update:
        self.up_atb1_6_6t5 = UPAttention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5, x_c=C_l7, k=2)
        
        self.up_atb1_5_5t4 = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c= C_l6, k=2)
        self.up_atb1_5_6t4 = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c=C_l7, k=4)
        
        self.up_atb1_4_4t3 = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l5, k=2)
        self.up_atb1_4_5t3 = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l6, k=4)
        self.up_atb1_4_6t3 = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c=C_l7, k=8)
        
        self.up_atb1_3_3t2 = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l4, k=2)
        self.up_atb1_3_4t2 = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l5, k=4)
        self.up_atb1_3_5t2 = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l6, k=8)
        self.up_atb1_3_6t2 = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c=C_l7, k=16)
        
        self.up_atb1_2_2t1 = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c=  C_l3, k=2)
        self.up_atb1_2_3t1 = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l4, k=4)
        self.up_atb1_2_4t1 = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l5, k=8)
        self.up_atb1_2_5t1 = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l6, k=16)
        self.up_atb1_2_6t1 = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c=C_l7, k=32)
        
        # 第二个编解码器使用的 ---------------------------------------
        self.atb2_5_1 = Attention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5)
        self.atb2_5_2 = Attention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5)
        self.atb2_4_1 = Attention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4)
        self.atb2_4_2 = Attention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4)
        self.atb2_3_1 = Attention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3)
        self.atb2_3_2 = Attention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3)
        self.atb2_2_1 = Attention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2)
        self.atb2_2_2 = Attention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2)
        self.atb2_1_1 = Attention_block(F_g= C_l2, F_l= C_l2, F_int= 16)
        self.atb2_1_2 = Attention_block(F_g= C_l2, F_l= C_l2, F_int= 16)
        # self.atb2_0 = Attention_block(F_g=  1, F_l=  1, F_int=  1)
        
        # up_rrb 系列通常在cat操作后，用于降低通道数。
        self.up_rrb2_5 = RRCNN_block(ch_in= C_l6*5, ch_out=C_l6, t=t)
        self.up_rrb2_4 = RRCNN_block(ch_in= C_l5*7, ch_out=C_l5, t=t)
        self.up_rrb2_3 = RRCNN_block(ch_in= C_l4*9, ch_out=C_l4, t=t)
        self.up_rrb2_2 = RRCNN_block(ch_in=  C_l3*11, ch_out= C_l3, t=t)
        self.up_rrb2_1 = RRCNN_block(ch_in=  C_l2*13, ch_out= C_l2, t=t)
        # self.up_rrb2_0 = RRCNN_block(ch_in=  C_l2, ch_out= C_l1, t=t)
        # self.up_rrb1_0 = RRCNN_block(ch_in=   1, ch_out=  1, t=t)
        
        self.up_atb2_6_6t5_1_x = UPAttention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5, x_c=C_l7, k=2)
        self.up_atb2_6_6t5_2_m = UPAttention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5, x_c=C_l7, k=2)
        
        self.up_atb2_5_5t4_1_x = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c= C_l6, k=2)
        self.up_atb2_5_6t4_1_x = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c=C_l7, k=4)
        self.up_atb2_5_5t4_2_m = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c= C_l6, k=2)
        self.up_atb2_5_6t4_2_m = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c=C_l7, k=4)
        
        self.up_atb2_4_4t3_1_x = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l5, k=2)
        self.up_atb2_4_5t3_1_x = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l6, k=4)
        self.up_atb2_4_6t3_1_x = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c=C_l7, k=8)
        self.up_atb2_4_4t3_2_m = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l5, k=2)
        self.up_atb2_4_5t3_2_m = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l6, k=4)
        self.up_atb2_4_6t3_2_m = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c=C_l7, k=8)
        
        self.up_atb2_3_3t2_1_x = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l4, k=2)
        self.up_atb2_3_4t2_1_x = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l5, k=4)
        self.up_atb2_3_5t2_1_x = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l6, k=8)
        self.up_atb2_3_6t2_1_x = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c=C_l7, k=16)
        self.up_atb2_3_3t2_2_m = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l4, k=2)
        self.up_atb2_3_4t2_2_m = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l5, k=4)
        self.up_atb2_3_5t2_2_m = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l6, k=8)
        self.up_atb2_3_6t2_2_m = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c=C_l7, k=16)
        
        self.up_atb2_2_2t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c=  C_l3, k=2)
        self.up_atb2_2_3t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c= C_l4, k=4)
        self.up_atb2_2_4t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c= C_l5, k=8)
        self.up_atb2_2_5t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c= C_l6, k=16)
        self.up_atb2_2_6t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c=C_l7, k=32)
        self.up_atb2_2_2t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c=  C_l3, k=2)
        self.up_atb2_2_3t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c= C_l4, k=4)
        self.up_atb2_2_4t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c= C_l5, k=8)
        self.up_atb2_2_5t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c= C_l6, k=16)
        self.up_atb2_2_6t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= C_l1, x_c=C_l7, k=32)
        
        # 第三个编解码器使用的 ---------------------------------------
        self.atb3_5_1 = Attention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5) #  x用
        self.atb3_5_2 = Attention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5) # mx用
        self.atb3_5_3 = Attention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5) # nx用 下同
        
        self.atb3_4_1 = Attention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4)
        self.atb3_4_2 = Attention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4)
        self.atb3_4_3 = Attention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4)
        
        self.atb3_3_1 = Attention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3)
        self.atb3_3_2 = Attention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3)
        self.atb3_3_3 = Attention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3)
        
        self.atb3_2_1 = Attention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2)
        self.atb3_2_2 = Attention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2)
        self.atb3_2_3 = Attention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2)
        
        self.atb3_1_1 = Attention_block(F_g= C_l2, F_l= C_l2, F_int= 16)
        self.atb3_1_2 = Attention_block(F_g= C_l2, F_l= C_l2, F_int= 16)
        self.atb3_1_3 = Attention_block(F_g= C_l2, F_l= C_l2, F_int= 16)
                
        # up_rrb 系列通常在cat操作后，用于降低通道数。
        self.up_rrb3_5 = RRCNN_block(ch_in= 7*C_l6, ch_out=C_l6, t=t)
        self.up_rrb3_4 = RRCNN_block(ch_in=10*C_l5, ch_out=C_l5, t=t)
        self.up_rrb3_3 = RRCNN_block(ch_in=13*C_l4, ch_out=C_l4, t=t)
        self.up_rrb3_2 = RRCNN_block(ch_in= 16*C_l3, ch_out= C_l3, t=t)
        self.up_rrb3_1 = RRCNN_block(ch_in= 19*C_l2, ch_out= C_l2, t=t)
        # self.up_rrb3_0 = RRCNN_block(ch_in=    C_l2, ch_out= C_l1, t=t)
        
        self.up_atb3_6_6t5_1_x = UPAttention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5, x_c=C_l7, k=2) #  x用
        self.up_atb3_6_6t5_2_m = UPAttention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5, x_c=C_l7, k=2) # mx用
        self.up_atb3_6_6t5_3_n = UPAttention_block(F_g=C_l6, F_l=C_l6, F_int=C_l5, x_c=C_l7, k=2) # nx用
        
        self.up_atb3_5_5t4_1_x = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c= C_l6, k=2) #  x用
        self.up_atb3_5_6t4_1_x = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c=C_l7, k=4) #  x用
        self.up_atb3_5_5t4_2_m = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c= C_l6, k=2) # mx用
        self.up_atb3_5_6t4_2_m = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c=C_l7, k=4) #  x用
        self.up_atb3_5_5t4_3_n = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c= C_l6, k=2) # nx用
        self.up_atb3_5_6t4_3_n = UPAttention_block(F_g=C_l5, F_l=C_l5, F_int=C_l4, x_c=C_l7, k=4) #  x用
        
        self.up_atb3_4_4t3_1_x = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l5, k=2)
        self.up_atb3_4_5t3_1_x = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l6, k=4)
        self.up_atb3_4_6t3_1_x = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c=C_l7, k=8)
        self.up_atb3_4_4t3_2_m = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l5, k=2)
        self.up_atb3_4_5t3_2_m = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l6, k=4)
        self.up_atb3_4_6t3_2_m = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c=C_l7, k=8)
        self.up_atb3_4_4t3_3_n = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l5, k=2)
        self.up_atb3_4_5t3_3_n = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c= C_l6, k=4)
        self.up_atb3_4_6t3_3_n = UPAttention_block(F_g=C_l4, F_l=C_l4, F_int= C_l3, x_c=C_l7, k=8)
        
        self.up_atb3_3_3t2_1_x = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l4, k=2)
        self.up_atb3_3_4t2_1_x = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l5, k=4)
        self.up_atb3_3_5t2_1_x = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l6, k=8)
        self.up_atb3_3_6t2_1_x = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c=C_l7, k=16)
        
        self.up_atb3_3_3t2_2_m = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l4, k=2)
        self.up_atb3_3_4t2_2_m = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l5, k=4)
        self.up_atb3_3_5t2_2_m = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l6, k=8)
        self.up_atb3_3_6t2_2_m = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c=C_l7, k=16)
        
        self.up_atb3_3_3t2_3_n = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l4, k=2)
        self.up_atb3_3_4t2_3_n = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l5, k=4)
        self.up_atb3_3_5t2_3_n = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c= C_l6, k=8)
        self.up_atb3_3_6t2_3_n = UPAttention_block(F_g= C_l3, F_l= C_l3, F_int= C_l2, x_c=C_l7, k=16)
        
        self.up_atb3_2_2t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c=  C_l3, k=2)
        self.up_atb3_2_3t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l4, k=4)
        self.up_atb3_2_4t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l5, k=8)
        self.up_atb3_2_5t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l6, k=16)
        self.up_atb3_2_6t1_1_x = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c=C_l7, k=32)
        
        self.up_atb3_2_2t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c=  C_l3, k=2)
        self.up_atb3_2_3t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l4, k=4)
        self.up_atb3_2_4t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l5, k=8)
        self.up_atb3_2_5t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l6, k=16)
        self.up_atb3_2_6t1_2_m = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c=C_l7, k=32)
        
        self.up_atb3_2_2t1_3_n = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c=  C_l3, k=2)
        self.up_atb3_2_3t1_3_n = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l4, k=4)
        self.up_atb3_2_4t1_3_n = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l5, k=8)
        self.up_atb3_2_5t1_3_n = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c= C_l6, k=16)
        self.up_atb3_2_6t1_3_n = UPAttention_block(F_g= C_l2, F_l= C_l2, F_int= 16, x_c=C_l7, k=32)

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
        
        x1 = self.rrb1_1(road1) #    1 256 256 ->   4 256 256
        x2 = self.dwConv1_1(x1) #   4 256 256 ->  4 128 128
        x2 = self.rrb1_2(x2)    #    4 128 128 ->   64 128 128
        x3 = self.dwConv1_2(x2) #   64 128 128 ->   64  64  64
        x3 = self.rrb1_3(x3)    #   64  64  64 ->  128  64  64
        x4 = self.dwConv1_3(x3) #  128  64  64 ->  128  32  32
        x4 = self.rrb1_4(x4)    #  128  32  32 ->  256  32  32
        x5 = self.dwConv1_4(x4) #  256  32  32 ->  256  16  16
        x5 = self.rrb1_5(x5)    #  256  16  16 ->  512  16  16
        x6 = self.dwConv1_5(x5) #  512  16  16 ->  512   8   8
        x6 = self.rrb1_6(x6)    #  512   8   8 -> 1024   8   8
        
        # # phase 1-6 decoding
        # data1_5 = self.upconv1_5(x6) # 1024 8 8 -> 512 16 16
        # x5_1 = self.atb1_5(g=data1_5,x=x5) # 512 16 16
        # tmp1_5 = torch.cat((x5_1,data1_5),dim=1) # 1024 16 16
        # data1_5 = self.up_rrb1_5(tmp1_5) # 1024 16 16 -> 512 16 16
        
        
        # phase 1-5 decoding
        data1_5 = self.upconv1_5(x6)                       # 1024  8  8 -> 512 16 16
        x5_1 = self.atb1_5(g=data1_5,x=x5)                 #  512 16 16
        updata1_5_1 = self.up_atb1_6_6t5(g=data1_5,x=x6)     #  512 16 16             2024年9月30日19:55:02 update
        tmp1_5 = torch.cat((x5_1,data1_5,updata1_5_1),dim=1) # 1024 16 16
        data1_5 = self.up_rrb1_5(tmp1_5)                   # 1024 16 16 -> 512 16 16
        
        # phase 1-4
        data1_4 = self.upconv1_4(data1_5)  # 512 16 16 -> 256 32 32
        x4_1 = self.atb1_4(g=data1_4,x=x4) # 256 32 32
        updata1_4_1 = self.up_atb1_5_5t4(g=data1_4,x=x5) # up_atb 会改变数据形状，从x5级别改为x4级别
        updata1_4_2 = self.up_atb1_5_6t4(g=data1_4,x=x6)
        tmp1_4 = torch.cat((x4_1,data1_4,updata1_4_1,updata1_4_2),dim=1) # 256*3 32 32
        data1_4 = self.up_rrb1_4(tmp1_4) # 256*3 32 32 -> 256 32 32
        
        # phase 1-3
        data1_3 = self.upconv1_3(data1_4) # 256 32 32 -> 128 64 64
        x3_1 = self.atb1_3(g=data1_3,x=x3) # 128 64 64
        updata1_3_1 = self.up_atb1_4_4t3(g=data1_3,x=x4) # 128 64 64
        updata1_3_2 = self.up_atb1_4_5t3(g=data1_3,x=x5) # 128 64 64
        updata1_3_3 = self.up_atb1_4_6t3(g=data1_3,x=x6) # 128 64 64
        tmp1_3 = torch.cat((x3_1,data1_3,updata1_3_1,updata1_3_2,updata1_3_3),dim=1) # 128*4 64 64
        data1_3 = self.up_rrb1_3(tmp1_3)
        
        # phase 1-2
        data1_2 = self.upconv1_2(data1_3) # 64 128 128
        x2_1 = self.atb1_2(g=data1_2,x=x2)
        updata1_2_1 = self.up_atb1_3_3t2(g=data1_2,x=x3)
        updata1_2_2 = self.up_atb1_3_4t2(g=data1_2,x=x4)
        updata1_2_3 = self.up_atb1_3_5t2(g=data1_2,x=x5)
        updata1_2_4 = self.up_atb1_3_6t2(g=data1_2,x=x6)
        tmp1_2 = torch.cat((x2_1,data1_2,updata1_2_1,updata1_2_2,updata1_2_3,updata1_2_4),dim=1)
        data1_2 = self.up_rrb1_2(tmp1_2)
        
        # phase 1-1
        data1_1 = self.upconv1_1(data1_2) # 32 256 256
        x1_1 = self.atb1_1(g=data1_1,x=x1)
        updata1_1_1 = self.up_atb1_2_2t1(g=data1_1,x=x2)
        updata1_1_2 = self.up_atb1_2_3t1(g=data1_1,x=x3)
        updata1_1_3 = self.up_atb1_2_4t1(g=data1_1,x=x4)
        updata1_1_4 = self.up_atb1_2_5t1(g=data1_1,x=x5)
        updata1_1_5 = self.up_atb1_2_6t1(g=data1_1,x=x6)
        tmp1_1 = torch.cat((x1_1,data1_1,updata1_1_1,updata1_1_2,updata1_1_3,updata1_1_4,updata1_1_5),dim=1)
        data1_1 = self.up_rrb1_1(tmp1_1) # 32 256 256
        
        # phase 1-0
        y_stage1 = self.conv1_0(data1_1)
        
        # encoding path
        # phase 2-encoding
        mx1 = self.rrb1_1(y_stage1) # 1 256 256 -> 3 256 256
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
        
        # phase 2-6 None
        
        # phase 2-5
        data2_5 = self.upconv2_5(mx6) # 512 16 16
        updata2_5_1 = self.up_atb2_6_6t5_1_x(g=data2_5,x= x6)
        updata2_5_2 = self.up_atb2_6_6t5_2_m(g=data2_5,x=mx6)
        mx5_1 = self.atb2_5_1(g=data2_5,x= x5)
        mx5_2 = self.atb2_5_2(g=data2_5,x=mx5)
        tmp2_5 = torch.cat((mx5_1,mx5_2,data2_5,updata2_5_1,updata2_5_2),dim=1)
        data2_5 = self.up_rrb2_5(tmp2_5)
        
        # phase 2-4
        data2_4 = self.upconv2_4(data2_5) # 256 32 32
        updata2_4_1_x = self.up_atb2_5_5t4_1_x(g=data2_4,x= x5)
        updata2_4_2_x = self.up_atb2_5_6t4_1_x(g=data2_4,x= x6)
        updata2_4_1_m = self.up_atb2_5_5t4_2_m(g=data2_4,x=mx5)
        updata2_4_2_m = self.up_atb2_5_6t4_1_x(g=data2_4,x=mx6)
        mx4_1 = self.atb2_4_1(g=data2_4,x=x4)
        mx4_2 = self.atb2_4_2(g=data2_4,x=mx4)
        tmp2_4 = torch.cat((mx4_1,mx4_2,data2_4,updata2_4_1_x,updata2_4_2_x,updata2_4_1_m,updata2_4_2_m),dim=1)
        data2_4 = self.up_rrb2_4(tmp2_4)
        
        # phase 2-3
        data2_3 = self.upconv2_3(data2_4) # 128 64 64
        updata2_3_1_x = self.up_atb2_4_4t3_1_x(g=data2_3,x= x4)
        updata2_3_2_x = self.up_atb2_4_5t3_1_x(g=data2_3,x= x5)
        updata2_3_3_x = self.up_atb2_4_6t3_1_x(g=data2_3,x= x6)
        updata2_3_1_m = self.up_atb2_4_4t3_2_m(g=data2_3,x=mx4)
        updata2_3_2_m = self.up_atb2_4_5t3_2_m(g=data2_3,x=mx5)
        updata2_3_3_m = self.up_atb2_4_6t3_2_m(g=data2_3,x=mx6)
        mx3_1 = self.atb2_3_1(g=data2_3,x=x3)
        mx3_2 = self.atb2_3_2(g=data2_3,x=mx3)
        tmp2_3 = torch.cat((mx3_1,mx3_2,data2_3,updata2_3_1_x,updata2_3_2_x,updata2_3_3_x,updata2_3_1_m,updata2_3_2_m,updata2_3_3_m),dim=1)
        data2_3 = self.up_rrb2_3(tmp2_3)
        
        # phase 2-2
        data2_2 = self.upconv2_2(data2_3) # 64 128 128
        updata2_2_1_x = self.up_atb2_3_3t2_1_x(g=data2_2,x= x3)
        updata2_2_2_x = self.up_atb2_3_4t2_1_x(g=data2_2,x= x4)
        updata2_2_3_x = self.up_atb2_3_5t2_1_x(g=data2_2,x= x5)
        updata2_2_4_x = self.up_atb2_3_6t2_1_x(g=data2_2,x= x6)
        updata2_2_1_m = self.up_atb2_3_3t2_2_m(g=data2_2,x=mx3)
        updata2_2_2_m = self.up_atb2_3_4t2_2_m(g=data2_2,x=mx4)
        updata2_2_3_m = self.up_atb2_3_5t2_2_m(g=data2_2,x=mx5)
        updata2_2_4_m = self.up_atb2_3_6t2_2_m(g=data2_2,x=mx6)
        mx2_1 = self.atb2_2_1(g=data2_2,x=x2)
        mx2_2 = self.atb2_2_2(g=data2_2,x=mx2)
        tmp2_2 = torch.cat((mx2_1,mx2_2,data2_2,updata2_2_1_x,updata2_2_2_x,updata2_2_3_x,updata2_2_4_x,updata2_2_1_m,updata2_2_2_m,updata2_2_3_m,updata2_2_4_m),dim=1)
        data2_2 = self.up_rrb2_2(tmp2_2)
        
        #phase 2-1
        data2_1 = self.upconv2_1(data2_2) # 32 256 256
        updata2_1_1_x = self.up_atb2_2_2t1_1_x(g=data2_1,x= x2)
        updata2_1_2_x = self.up_atb2_2_3t1_1_x(g=data2_1,x= x3)
        updata2_1_3_x = self.up_atb2_2_4t1_1_x(g=data2_1,x= x4)
        updata2_1_4_x = self.up_atb2_2_5t1_1_x(g=data2_1,x= x5)
        updata2_1_5_x = self.up_atb2_2_6t1_1_x(g=data2_1,x= x6)
        updata2_1_1_m = self.up_atb2_2_2t1_2_m(g=data2_1,x=mx2)
        updata2_1_2_m = self.up_atb2_2_3t1_2_m(g=data2_1,x=mx3)
        updata2_1_3_m = self.up_atb2_2_4t1_2_m(g=data2_1,x=mx4)
        updata2_1_4_m = self.up_atb2_2_5t1_2_m(g=data2_1,x=mx5)
        updata2_1_5_m = self.up_atb2_2_6t1_2_m(g=data2_1,x=mx6)
        mx2_1 = self.atb2_1_1(g=data2_1,x= x1)
        mx2_2 = self.atb2_1_1(g=data2_1,x=mx1)
        tmp2_1 = torch.cat((mx2_1,mx2_2,data2_1,updata2_1_1_x,updata2_1_2_x,updata2_1_3_x,updata2_1_4_x,updata2_1_5_x,updata2_1_1_m,updata2_1_2_m,updata2_1_3_m,updata2_1_4_m,updata2_1_5_m), dim=1)
        data2_1 = self.up_rrb2_1(tmp2_1)
        
        #phase 2-0
        y_stage2 = self.conv2_0(data2_1)
        
        # encoding path
        # phase 3-encoding
        nx1 = self.rrb3_1(y_stage2) # 1 256 256 -> 3 256 256
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
        updata3_5_1_x = self.up_atb3_6_6t5_1_x(g=data3_5,x= x6)
        updata3_5_1_m = self.up_atb3_6_6t5_2_m(g=data3_5,x=mx6)
        updata3_5_1_n = self.up_atb3_6_6t5_3_n(g=data3_5,x=nx6)
        nx5_1_x = self.atb3_5_1(g=data3_5,x= x5)
        nx5_2_m = self.atb3_5_2(g=data3_5,x=mx5)
        nx5_3_n = self.atb3_5_3(g=data3_5,x=nx5)
        tmp3_5 = torch.cat((nx5_1_x,nx5_2_m,nx5_3_n,data3_5,updata3_5_1_x,updata3_5_1_m,updata3_5_1_n),dim=1)
        data3_5 = self.up_rrb3_5(tmp3_5)
        # phase 3-4
        data3_4 = self.upconv3_4(data3_5) # 256 32 32
        updata3_4_1_x = self.up_atb3_5_5t4_1_x(g=data3_4,x= x5)
        updata3_4_2_x = self.up_atb3_5_6t4_1_x(g=data3_4,x= x6)
        updata3_4_1_m = self.up_atb3_5_5t4_2_m(g=data3_4,x=mx5)
        updata3_4_2_m = self.up_atb3_5_6t4_2_m(g=data3_4,x=mx6)
        updata3_4_1_n = self.up_atb3_5_5t4_3_n(g=data3_4,x=nx5)
        updata3_4_2_n = self.up_atb3_5_6t4_3_n(g=data3_4,x=mx6)
        nx4_1 = self.atb3_4_1(g=data3_4,x= x4)
        nx4_2 = self.atb3_4_2(g=data3_4,x=mx4)
        nx4_3 = self.atb3_4_3(g=data3_4,x=nx4)
        tmp3_4 = torch.cat((nx4_1,nx4_2,nx4_3,data3_4,updata3_4_1_x,updata3_4_2_x,updata3_4_1_m,updata3_4_2_m,updata3_4_1_n,updata3_4_2_n),dim=1)
        data3_4 = self.up_rrb3_4(tmp3_4)
        # phase 3-3
        data3_3 = self.upconv3_3(data3_4) # 128 64 64
        updata3_3_1_x = self.up_atb3_4_4t3_1_x(g=data3_3,x= x4)
        updata3_3_2_x = self.up_atb3_4_5t3_1_x(g=data3_3,x= x5)
        updata3_3_3_x = self.up_atb3_4_6t3_1_x(g=data3_3,x= x6)
        updata3_3_1_m = self.up_atb3_4_4t3_2_m(g=data3_3,x=mx4)
        updata3_3_2_m = self.up_atb3_4_5t3_2_m(g=data3_3,x=mx5)
        updata3_3_3_m = self.up_atb3_4_6t3_2_m(g=data3_3,x=mx6)
        updata3_3_1_n = self.up_atb3_4_4t3_3_n(g=data3_3,x=nx4)
        updata3_3_2_n = self.up_atb3_4_5t3_3_n(g=data3_3,x=nx5)
        updata3_3_3_n = self.up_atb3_4_6t3_3_n(g=data3_3,x=nx6)
        nx3_1 = self.atb3_3_1(g=data3_3,x= x3)
        nx3_2 = self.atb3_3_1(g=data3_3,x=mx3)
        nx3_3 = self.atb3_3_1(g=data3_3,x=nx3)
        tmp3_3 = torch.cat((nx3_1,nx3_2,nx3_3,data3_3,updata3_3_1_x,updata3_3_2_x,updata3_3_3_x,updata3_3_1_m,updata3_3_2_m,updata3_3_3_m,updata3_3_1_n,updata3_3_2_n,updata3_3_3_n),dim=1)
        data3_3 = self.up_rrb3_3(tmp3_3)
        
        # phase 3-2
        data3_2 = self.upconv3_2(data3_3) # 64 128 128
        updata3_2_1_x = self.up_atb3_3_3t2_1_x(g=data3_2,x= x3)
        updata3_2_2_x = self.up_atb3_3_4t2_1_x(g=data3_2,x= x4)
        updata3_2_3_x = self.up_atb3_3_5t2_1_x(g=data3_2,x= x5)
        updata3_2_4_x = self.up_atb3_3_6t2_1_x(g=data3_2,x= x6)
        
        updata3_2_1_m = self.up_atb3_3_3t2_2_m(g=data3_2,x=mx3)
        updata3_2_2_m = self.up_atb3_3_4t2_2_m(g=data3_2,x=mx4)
        updata3_2_3_m = self.up_atb3_3_5t2_2_m(g=data3_2,x=mx5)
        updata3_2_4_m = self.up_atb3_3_6t2_2_m(g=data3_2,x=mx6)
        
        updata3_2_1_n = self.up_atb3_3_3t2_3_n(g=data3_2,x=nx3)
        updata3_2_2_n = self.up_atb3_3_4t2_3_n(g=data3_2,x=nx4)
        updata3_2_3_n = self.up_atb3_3_5t2_3_n(g=data3_2,x=nx5)
        updata3_2_4_n = self.up_atb3_3_6t2_3_n(g=data3_2,x=nx6)
        
        nx2_1 = self.atb3_2_1(g=data3_2,x= x2)
        nx2_2 = self.atb3_2_2(g=data3_2,x=mx2)
        nx2_3 = self.atb3_2_3(g=data3_2,x=nx2)
        tmp3_2 = torch.cat((nx2_1,nx2_2,nx2_3,data3_2,updata3_2_1_x,updata3_2_2_x,updata3_2_3_x,updata3_2_4_x,updata3_2_1_m,updata3_2_2_m,updata3_2_3_m,updata3_2_4_m,updata3_2_1_n,updata3_2_2_n,updata3_2_3_n,updata3_2_4_n),dim=1)
        data3_2 = self.up_rrb3_2(tmp3_2)
        # phase 3-1
        data3_1 = self.upconv3_1(data3_2) # 32 256 256
        updata3_1_1_x = self.up_atb3_2_2t1_1_x(g=data3_1,x= x2)
        updata3_1_2_x = self.up_atb3_2_3t1_1_x(g=data3_1,x= x3)
        updata3_1_3_x = self.up_atb3_2_4t1_1_x(g=data3_1,x= x4)
        updata3_1_4_x = self.up_atb3_2_5t1_1_x(g=data3_1,x= x5)
        updata3_1_5_x = self.up_atb3_2_6t1_1_x(g=data3_1,x= x6)
        
        updata3_1_1_m = self.up_atb3_2_2t1_2_m(g=data3_1,x=mx2)
        updata3_1_2_m = self.up_atb3_2_3t1_2_m(g=data3_1,x=mx3)
        updata3_1_3_m = self.up_atb3_2_4t1_2_m(g=data3_1,x=mx4)
        updata3_1_4_m = self.up_atb3_2_5t1_2_m(g=data3_1,x=mx5)
        updata3_1_5_m = self.up_atb3_2_6t1_2_m(g=data3_1,x=mx6)
        
        updata3_1_1_n = self.up_atb3_2_2t1_3_n(g=data3_1,x=nx2)
        updata3_1_2_n = self.up_atb3_2_3t1_3_n(g=data3_1,x=nx3)
        updata3_1_3_n = self.up_atb3_2_4t1_3_n(g=data3_1,x=nx4)
        updata3_1_4_n = self.up_atb3_2_5t1_3_n(g=data3_1,x=nx5)
        updata3_1_5_n = self.up_atb3_2_6t1_3_n(g=data3_1,x=nx6)
        nx1_1 = self.atb3_1_1(g=data3_1,x= x1)
        nx1_2 = self.atb3_1_2(g=data3_1,x=mx1)
        nx1_3 = self.atb3_1_3(g=data3_1,x=nx1)
        tmp3_1 = torch.cat((nx1_1,nx1_2,nx1_3,data3_1,updata3_1_1_x,updata3_1_2_x,updata3_1_3_x,updata3_1_4_x,updata3_1_5_x,updata3_1_1_m,updata3_1_2_m,updata3_1_3_m,updata3_1_4_m,updata3_1_5_m,updata3_1_1_n,updata3_1_2_n,updata3_1_3_n,updata3_1_4_n,updata3_1_5_n),dim=1)
        data3_1 = self.up_rrb3_1(tmp3_1)
        # phase 3-0
        y_pred = self.conv3_0(data3_1)

        return y_pred,y_stage2,y_stage1