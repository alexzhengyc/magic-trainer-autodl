from Dreambooth import Dreambooth
configs = [

    ## maximal input
    {
        "dir_name": "chenweiting-512-200-8bit-hassan",
        "train_data": "test_train",
        "reg_data": "test_reg",
        "resolution": 512,
        "v2" : False,
        "sd_path": "/root/autodl-tmp/models/Stable-diffusion/hassanblend14.safetensors",
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
    # ,
    ## minimal input
    {
        "dir_name": "chenweiting-512-10-Ada-hassan",
        "train_data": "test_train",
        "reg_data": "test_reg",
        "train_repeats": 10,
        "reg_repeats": 1,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
        ],

    },
] 



for config in configs:
    model = Dreambooth(**config)

    # model.prepare(data_anotation = "blip") # @param ["none", "waifu", "blip", "combined"]

    model.train()