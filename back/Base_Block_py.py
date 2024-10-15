import torch.nn as nn
# import torch.nn.functional as F


class up_conv(nn.Module):
    """ 将当前的长宽像素数各扩大至2倍，通道自定。通常会将通道数降低为原来的一半。 """
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_bn_relu(nn.Module):
    """ 常见的conv_bn_relu块 默认kernel_size=3, stride=1, padding=1, bias=True\n
    常见的conv1x1的kernel_size=1, stride=1, padding=0"""
    def __init__(self, ch_in, ch_out,kernel=3,stride=1,padding=1,bias=True):
        super(conv_bn_relu, self).__init__()
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convbnrelu(x)
        return x

 
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            #(h-3+2*1)/1+1=h所以最终输入和输出是一样的
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
 
            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1
    
class RRCNN_block(nn.Module):
    """ 自定义输入和输出的通道数，参数t为循环次数。\n 
    默认循环2次，包含Conv,BN,ReLU3个操作。\n
    后一个操作会将链接最初的输入和中间的输出一起操作。"""
    def __init__(self, ch_in, ch_out, t=2):
        """ ch_in : input data channels
        ch_out : output data channels
        t : loop times. 
        this block have 2 Recurrent_block(Rb). so if t equals 2, each Rb will repeat 2 times. result is 2 by 2 equal 4 times.
        """
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """ F_g 是作为注意力的数据的输入通道数，\n
            F_l 是作为主要数据的输入通道数，\n
            F_int 是注意力和主要数据卷积后的通道数（压缩或膨胀后，通常压缩到输入通道数的一半）。\n
            不会改变数据的形态。
        """
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class UPAttention_block(nn.Module):
    r""" class: 
    F_g 为输入的g的通道数，F_l为输入的x通道数x_c，经过通过一次转换后的通道数。F_int为中转通道数，g和d都会卷积到该中转通道数，最后形成注意力矩阵\n
    k为x进行ConvTranspose2d操作时的卷积核大小和步进大小。\n
    最后输出的尺寸应为[b,F_l,w*k,h*k],其中w*k和h*k应与输入的g的w和h相等。
    """
    def __init__(self, F_g, F_l, F_int,x_c,k):
        r""" init:
        F_g 为输入的g的通道数，F_l为输入的x通道数x_c，经过通过一次转换后的通道数。F_int为中转通道数，g和d都会卷积到该中转通道数，最后形成注意力矩阵\n
        k为x进行ConvTranspose2d操作时的卷积核大小和步进大小。\n
        最后输出的尺寸应为[b,F_l,w*k,h*k],其中w*k和h*k应与输入的g的w和h相等。
        """
        super(UPAttention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.change=nn.ConvTranspose2d(x_c, F_l, kernel_size=k, stride=k, padding=0, bias=False)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        x=self.change(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi