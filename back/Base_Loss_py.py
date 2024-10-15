import torch
import torch.nn as nn
from back.Base_Method_py import de_norm,limit_zero2one

class BrightnessDarknessLossL2(nn.Module):
    def __init__(self,alpha=100,y_black_factor=100,y_gray_factor=100,y_white_factor=100,y_brightness_factor=100,x_black_factor=1):
        super(BrightnessDarknessLossL2, self).__init__()
        self.alpha = alpha
        self.y_black_factor = y_black_factor # [~,0.002]区域的敏感度
        self.y_gray_factor = y_gray_factor # [0.1,0.32]区域的敏感度
        self.y_white_factor = y_white_factor # [0.9-~]区域的敏感度
        self.y_brightness_factor = y_brightness_factor
        self.x_black_factor = x_black_factor
        self.y_black_top_limit = 0.002
        self.y_gray_bottom_limit = 0.01
        self.y_gray_top_limit = 0.32
        self.y_white_bottom_limit = 0.9
        

    def forward(self, pred_y, y, x):
        """ pred_y, y, x 应均为[B,1,H,W] """
        pred_y = de_norm(pred_y)
        y=de_norm(y)
        x=de_norm(x)
        """ 以上操作将值域化为[0,1] """
        
        # 亮度权重
        # brightness_weights = self.y_brightness_factor * y / (torch.max(y) + 1e-8)
        brightness_weights = self.y_brightness_factor
        # 忽略黑色区域的权重
        # ignore_mask_x = (x <= -0.98).float()    # 为原图x中的黑色区域
        # ignore_mask_y_outer = (y <= 0.02).float() * (torch.sum(y, dim=(1, 2, 3), keepdim=True) > 1000) # 为目标图像y的外围黑色部分
        
        # 有效黑色区域权重
        valid_mask_y_black = (y < self.y_black_top_limit).float() # 识别y中的黑色区域 颜色在[~,black_top_limit]之间的区域
        black_sensitive_weights = valid_mask_y_black * self.y_black_factor
        
        valid_mask_y_gray = (y >= self.y_gray_bottom_limit).float() # 识别y中的灰色黑色区域 颜色在[y_gray_bottom_limit,y_gray_top_limit]之间的区域
        gray_sensitive_weights = (y <= self.y_gray_top_limit).float() * valid_mask_y_gray * self.y_gray_factor
        
        valid_mask_y_white = (y >= self.y_white_bottom_limit).float() # 识别y中的白色区域 颜色在[y_white_bottom_limit,~]之间的区域
        white_sensitive_weights = valid_mask_y_white * self.y_white_factor
    
        # 计算损失
        loss = ((pred_y - y)**2 * (1 + brightness_weights + black_sensitive_weights + gray_sensitive_weights + white_sensitive_weights)).mean()
        return loss * self.alpha
    
class BrightnessDarknessLossL1(nn.Module):
    def __init__(self,alpha=100,y_black_factor=100,y_gray_factor=100,y_white_factor=100,y_brightness_factor=100,x_black_factor=1):
        super(BrightnessDarknessLossL1, self).__init__()
        self.alpha = alpha
        self.y_black_factor = y_black_factor # [~,0.002]区域的敏感度
        self.y_gray_factor = y_gray_factor # [0.1,0.32]区域的敏感度
        self.y_white_factor = y_white_factor # [0.9-~]区域的敏感度
        self.y_brightness_factor = y_brightness_factor
        self.x_black_factor = x_black_factor
        self.y_black_top_limit = 0.002
        self.y_gray_bottom_limit = 0.01
        self.y_gray_top_limit = 0.32
        self.y_white_bottom_limit = 0.9
        

    def forward(self, pred_y, y, x):
        """ pred_y, y, x 应均为[B,1,H,W] """
        pred_y = de_norm(pred_y)
        y=de_norm(y)
        x=de_norm(x)
        """ 以上操作将值域化为[0,1] """
        
        # 亮度权重
        brightness_weights = self.y_brightness_factor * y / (torch.max(y) + 1e-8)
        # 忽略黑色区域的权重
        # ignore_mask_x = (x <= -0.98).float()    # 为原图x中的黑色区域
        # ignore_mask_y_outer = (y <= 0.02).float() * (torch.sum(y, dim=(1, 2, 3), keepdim=True) > 1000) # 为目标图像y的外围黑色部分
        
        # 有效黑色区域权重
        valid_mask_y_black = (y> self.y_black_top_limit).float() # 识别y中的黑色区域 颜色在[~,black_top_limit]之间的区域
        black_sensitive_weights = valid_mask_y_black * self.y_black_factor
        
        valid_mask_y_gray = (y >= self.y_gray_bottom_limit).float() # 识别y中的灰色黑色区域 颜色在[y_gray_bottom_limit,y_gray_top_limit]之间的区域
        gray_sensitive_weights = (y <= self.y_gray_top_limit).float() * valid_mask_y_gray * self.y_gray_factor
        
        valid_mask_y_white = (y >= self.y_white_bottom_limit).float() # 识别y中的白色区域 颜色在[y_white_bottom_limit,~]之间的区域
        white_sensitive_weights = valid_mask_y_white * self.y_white_factor
    
        # 计算损失
        loss = (torch.abs(pred_y - y) * (1 + brightness_weights + black_sensitive_weights + gray_sensitive_weights + white_sensitive_weights)).mean()
        return loss * self.alpha