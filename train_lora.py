from Lora import Lora

configs = [

    ## maximal input
    {
        "dir_name": "cwt-10-lion1-1.5",
        "train_data": "alex-768",                ## 将数据集放在 autodl-tmp/dataset下， 填cwt则表示instance dir为 autodl-tmp/dataset/cwt
        "reg_data": "",                      ## 将数据集放在 autodl-tmp/dataset下， 填man则表示class dir为 autodl-tmp/dataset/man                   
        "resolution": 768,
        "v2" : False,
        # "sd_path": "/root/autodl-tmp/models/Stable-diffusion/hassanblend14.safetensors",
        "sd_path": "/root/autodl-tmp/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
        "instance_token": "chenweiting man",     ## 用于自动生成标题时使用
        "class_token": "man",                    ## class image 统一的标题
        "train_repeats": 10,
        "reg_repeats": 1,
        "num_epochs": 1,
        "network_dim": 128,
        "network_alpha": 64,
        "train_batch_size": 4,
        "optimizer_type": "Lion", # @param ["AdamW", "AdamW8bit", "Lion", "DAdaptation", "AdaFactor", "SGDNesterov", "SGDNesterov8bit"]
        "unet_lr": 1.0e-6,
        "text_encoder_lr": 0.5e-6,
        "lr_scheduler" : "polynomial",  #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
        "prior_loss_weight": 1,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
            "1 chenweiting man with black hat",
        ],
        "images_per_prompt": 2,
        "sample_n_epoch_ratio" : 0.5,

    }
    ,{
        "dir_name": "cwt-10-lion1-hassan",
        "train_data": "chenweiting-512",
        "reg_data": "",
        "resolution": 512,
        "v2" : False,
        "sd_path": "/root/autodl-tmp/models/Stable-diffusion/hassanblend14.safetensors",
        # "sd_path": "/root/autodl-tmp/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
        "instance_token": "chenweiting man", 
        "class_token": "man",
        "train_repeats": 10,
        "reg_repeats": 1,
        "num_epochs": 1,
        "network_dim": 128,
        "network_alpha": 64,
        "train_batch_size": 4,
        "optimizer_type": "Lion", # @param ["AdamW", "AdamW8bit", "Lion", "DAdaptation", "AdaFactor", "SGDNesterov", "SGDNesterov8bit"]
        "unet_lr": 1.0e-6,
        "text_encoder_lr": 0.5e-6,
        "lr_scheduler" : "polynomial",  #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
        "prior_loss_weight": 1,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
            "1 chenweiting man with black hat",
        ],
        "images_per_prompt": 2,
        "sample_n_epoch_ratio" : 0.5,

    }
] 

for config in configs:
    model = Lora(**config)
    model.train()