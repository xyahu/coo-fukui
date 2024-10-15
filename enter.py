import torch
import back.config
import sys

def start_train():
    # 初始化配置参数
    config = back.config.Config()
    print(f"{config.time_stamp} --- start...")
    # patch = (1, config.opt.H // 2 ** 4, config.opt.W // 2 ** 4) # 1,16,16
    
    
    # 初始化训练模型
    from models.UDCTS_GAN_py import UDCTS_GAN
    model = UDCTS_GAN()
    model.init(config)
    model = config.adapt_cuda(model)
    
    model.save_model_info()
    model.init_config_dataset()
    model.init_config_optimizer()
    model.save_model_info()
    
    for epoch in range(0,200):
        model.train_step(epoch)
        model.train_sample(epoch)
        model.eval_step(epoch)
        model.eval_sample(epoch)
    pass


if __name__ == "__main__":
    
    # 初始化配置参数
    config = back.config.Config()
    print(f"{config.time_stamp} --- start...")
    # patch = (1, config.opt.H // 2 ** 4, config.opt.W // 2 ** 4) # 1,16,16
    
    
    # 初始化训练模型
    from models.GAN_UpwardDense_py import GAN_UpwardDense
    # from models.DenseDoubleUnetModel_py import DenseDoubleUnet
    model = GAN_UpwardDense()
    model.init(config)
    model = config.adapt_cuda(model)
    
    model.save_model_info()
    model.init_config_dataset()
    model.init_config_optimizer()
    model.save_model_info()
    
    model.train_s()
    # TODO :#
    print(f"Task: {config.time_stamp} was completed in {config.now()}!")
    pass