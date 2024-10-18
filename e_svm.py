import torch
import torch.nn as nn
from back.Base_Model_Vector_py import Base_Model_Vector
from back.config import Config

class SVM(nn.Module):
    def __init__(self,HWC):
        super(SVM, self).__init__()
        H = HWC[0]
        W = HWC[1]
        C = HWC[2]
        self.H = H
        self.W = W
        self.C = C
        # 定义线性层，将输入展开为一维向量后映射到输出
        self.linear = nn.Linear(C * H * W, 1)

    def forward(self, x):
        # 展开输入
        x = x.view(-1, self.C * self.H * self.W)
        output = self.linear(x)
        return output

class SVM_Vector(Base_Model_Vector): 
    def __init__(self, c, m):
        super(SVM_Vector,self).__init__(c, m)
        self.ct_400 = 'A-train'
        self.ct_64 =  'A-val'
        self.pet_400 = 'B-train'
        self.pet_64 = 'B-val'
        
    def compute_loss(self, y_pred, y_truth, x_input=None):
        """ SVM Vector use hinge loss to optimize SVM. """
        return torch.mean(torch.clamp(1 - y_pred.squeeze() * y_truth, min=0))
    
if __name__ == "__main__":
    c = Config()
    HWC = (c.opt.H,c.opt.W,c.opt.C)
    m = SVM_Vector(c,SVM(HWC))
    m = c.adapt_cuda(m)
    m.init(c)
    # m.init_config_dataset(A_train=m.pet_64,B_train=m.ct_64,A_val = m.pet_400,B_val=m.ct_400)
    m.init_config_dataset(A_train=m.pet_400,B_train=m.ct_400,A_val = m.pet_64,B_val=m.ct_64)

    # m.print_model_info()
    m.save_model_info()
    
    m.train_s()
    
    
    pass