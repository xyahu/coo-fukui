import torch
import torch.cuda
import datetime

from models.G_UDCTS_Generator_py import UDCTS_Generator
from models.G_L7_Generator_py import L7_UpwardDense_G
from models.G_L7_3_128_Generator_py import L7_Template_UpwardDense_G

class Tester:
    def __init__(self) -> None:
        # 假设我们有GPU，并选择第一个GPU
        # self.device = "cpu"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        str2time_param = '%Y_%m_%d_%H_%M_%S'
        print(f"device:{self.device}, type:{type(self.device)}")
        pass

    def model_test(self,model):
        time1 = datetime.datetime.now()
        print(f"------[{model.model_name}] IO test start------[{time1}]")
        if self.device != torch.device("cpu"):
            model.cuda(self.device)
        
        model.display_model_info()
        
        # 创建一个形状为B=2, C=3, H=224, W=224的随机Tensor
        x_input = torch.randn(2, 3, 256, 256, device=self.device)
        print(f"--------> Input shape is [{x_input.shape}]")

        y_pred,y_stage2,y_stage1 = model(x_input)
        print(f"-------->Output shape is [{y_pred.shape}]")
        time2 = datetime.datetime.now()
        print(f"------[{model.model_name}] IO test  end ------[{time2}]")
        print(f"Time Cost is {self.get_time_duration(time1,time2)}")
        pass
    
    def compare_params(self,m2,m1):
        compare_value = m2.count_parameters(m2) - m1.count_parameters(m1)
        print(f"{m2.model_name}.params - {m1.model_name}.params : {m2.count_parameters(m2)} - {m1.count_parameters(m1)} = {compare_value}")
        pass
    
    def get_new_time_stamp(self):
        current_datetime = datetime.datetime.now()
        auto_time_stamp = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        return auto_time_stamp
    
    def get_time_duration(self,prev,now):
        dura = now - prev
        # dura_str = dura.strftime('%Y_%m_%d_%H_%M_%S')
        return dura

def test1():
    tester = Tester()
    m1 = L7_UpwardDense_G()
    m2 = UDCTS_Generator()
    m3 = L7_Template_UpwardDense_G()
    tester.model_test(m1)
    tester.model_test(m2)
    tester.model_test(m3)
    
    # L7_UpwardDense_G == L7_Template_UpwardDense_G
    
    tester.compare_params(m2,m3)
    
    pass

if __name__ == "__main__":
    t = Tester()
    
    from models.G_Template_UpwardDense_py import G_Template_UpwardDense
    # m1 = L7_UpwardDense_G()
    # m2 = L7_Template_UpwardDense_G()
    m0 = G_Template_UpwardDense()
    
    t.model_test(m0)
    
    # t.model_test(m1)
    # t.model_test(m1)
    # t.compare_params(m2,m1)
    pass