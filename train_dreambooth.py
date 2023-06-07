from Dreambooth import Dreambooth
configs = [

    ## maximal input
    {
        "dir_name": "chenweiting-512-200-8bit-hassan",
        "train_data": "test1",                   ## 将数据集放在 autodl-tmp/dataset下， 填cwt则表示instance dir为 autodl-tmp/dataset/cwt
        "reg_data": "",                          ## 将数据集放在 autodl-tmp/dataset下， 填man则表示class dir为 autodl-tmp/dataset/man  
        "reg_data": "",
        "resolution": 512,
        "v2" : False,
        # "sd_path": "/root/autodl-tmp/models/Stable-diffusion/HassanBlend1.4_Safe.safetensors",
        "sd_path": "/root/autodl-tmp/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
        "instance_token": "chenweiting man", 
        "class_token": "man",
        "train_repeats": 10,
        "reg_repeats": 1,
        "max_train_steps" : 200,
        "train_batch_size": 1,
        "optimizer_type": "AdamW8bit", # @param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
        "learning_rate" : 1e-6,
        "lr_scheduler" : "polynomial",  #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
        "prior_loss_weight": 1.0,
        "sample_every_n_steps" : 100,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
        ],
        "images_per_prompt": 1,
        "save_n_epoch_ratio" : 0.5,
    }
] 



for config in configs:
    model = Dreambooth(**config)
    model.train()