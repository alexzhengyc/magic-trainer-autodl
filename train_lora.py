from Lora import Lora
configs = [

    ## maximal input
    {
        "dir_name": "x1",
        "train_data": "test_train",
        "reg_data": "test_reg",
        "resolution": 512,
        "v2" : False,
        "sd_path": "/root/autodl-tmp/models/Stable-diffusion/hassanblend14.safetensors",
        "instance_token": "chenweiting man", 
        "class_token": "man",
        "train_repeats": 2,
        "reg_repeats": 1,
        "num_epochs": 1,
        "network_dim": 128,
        "network_alpha": 64,
        "train_batch_size": 1,
        "optimizer_type": "DAdaptation", # @param ["AdamW", "AdamW8bit", "Lion", "DAdaptation", "AdaFactor", "SGDNesterov", "SGDNesterov8bit"]
        "unet_lr": 1.0,
        "text_encoder_lr": 1,
        "lr_scheduler" : "polynomial",  #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
        "prior_loss_weight": 1,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
        ],
        "images_per_prompt": 1,
        "save_n_epochs_ratio" : 0.5,

    }
    # ,
    # ## minimal input
    ,{
        "dir_name": "x2",
        "train_data": "",
        "reg_data": "",
        "train_repeats": 1,
        "reg_repeats": 1,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
        ],

    }
] 

for config in configs:
    model = Lora(**config)

    # model.prepare(data_anotation = "blip")  # @param ["none", "waifu", "blip", "combined"]
    
    model.train()