import back.config

if __name__ == "__main__":
    
    # 初始化配置参数
    config = back.config.Config()
    print(f"{config.time_stamp} --- start...")
    patch = (1, config.opt.H // 2 ** 4, config.opt.W // 2 ** 4) # 1,16,16
    
    
    # 初始化训练模型
    from back.UDCTS_GAN_py import UDCTS_GAN
    model = UDCTS_GAN()
    model.init(config)
    model = config.adapt_cuda(model)
    model.init_config_dataset()
    model.init_config_optimizer()
    
    model.save_model_info()
    model.train_s()
    # TODO :#
    print(f"{config.now()} done!")
    pass