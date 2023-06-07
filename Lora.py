#!/usr/bin/env python
# coding: utf-8
import os
import sys
import shutil
import random
from typing import Any
import toml
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import torch
import subprocess
from accelerate.utils import write_basic_config

# define class Dreambooth
class Lora():
    def __init__(self, **kwargs):
        
        self.dir_name = kwargs.get("dir_name", "default")
        self.train_data = kwargs.get("train_data", "")
        self.reg_data = kwargs.get("reg_data", "")
        self.sd_path = kwargs.get("sd_path", "/root/autodl-tmp/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors")
        self.v2 = kwargs.get("v2", False)
        self.vae_path = kwargs.get("vae", "/root/autodl-tmp/models/VAE/vae-ft-mse-840000-ema-pruned.ckpt")        
        self.instance_token = kwargs.get("instance_token", "")
        self.class_token = kwargs.get("class_token", "")
        self.train_repeats = kwargs.get("train_repeats", 10)
        self.reg_repeats = kwargs.get("reg_repeats", 1)
        self.num_epochs = kwargs.get("num_epochs", 1)
        self.network_dim = kwargs.get("network_dim", 64)
        self.network_alpha = kwargs.get("network_alpha", 32)
        self.train_unet = kwargs.get("train_unet", True)
        self.train_text_encoder = kwargs.get("train_text_encoder", True)
        self.unet_lr = kwargs.get("unet_lr", 1.0)
        self.text_encoder_lr = kwargs.get("text_encoder_lr", self.unet_lr/2)
        self.prompts = kwargs.get("prompts", None)
        self.images_per_prompt = kwargs.get("images_per_prompt", 1)
        self.optimizer_type = kwargs.get("optimizer_type", "DAdaptation")
        self.prior_loss_weight = kwargs.get("prior_loss_weight", 1.0)
        self.resolution = kwargs.get("resolution", 512)
        self.save_every_n_epochs = kwargs.get("save_every_n_epochs")
        self.sample_n_epoch_ratio = kwargs.get("sample_n_epoch_ratio")
        self.train_batch_size = kwargs.get("train_batch_size", 4)
        self.lr_scheduler = kwargs.get("lr_scheduler", "polynomial")
        self.flip_aug = kwargs.get("flip_aug", False)

        self.project_name = self.dir_name
        self.root_dir = "/root/magic-trainer-autodl"
        self.output_dir = "/root/autodl-tmp/training"
        self.dataset_dir = "/root/autodl-tmp/dataset"
        self.save_model_dir = "/root/autodl-tmp/models/Lora"

        self.pre = "masterpiece, best quality" 
        self.negative = "lowres, blurry" 
        self.sampler = "k_dpm_2"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
        self.scale = 7
        self.steps = 28  # @param {type: "slider", min: 1, max: 100}
        self.width = 512  # @param {type: "integer"}
        self.height = 512  # @param {type: "integer"}
        self.keep_tokens = 0
        self.caption_extension = ".txt"
        self.lowram = False
        self.v_parameterization = False

        self.caption_dropout_rate = 0
        self.caption_dropout_every_n_epochs = 0

        self.repo_dir = os.path.join(self.root_dir, "kohya_ss_revised")
        self.training_dir = os.path.join(self.output_dir, self.dir_name)
        self.train_data_dir = os.path.join(self.training_dir, "train_data")
        self.reg_data_dir = os.path.join(self.training_dir, "reg_data")
        self.config_dir = os.path.join(self.training_dir, "config")
        self.accelerate_config = os.path.join(self.repo_dir, "accelerate_config/config.yaml")
        self.tools_dir = os.path.join(self.repo_dir, "tools")
        self.finetune_dir = os.path.join(self.repo_dir, "finetune")
        self.sample_dir = os.path.join(self.training_dir, "sample")
        self.logging_dir = os.path.join(self.training_dir, "log")

        for dir in [
            self.training_dir,
            self.train_data_dir,
            self.reg_data_dir,
            self.config_dir,
            self.output_dir, 
            self.sample_dir,
            ]:
            os.makedirs(dir, exist_ok=True)

        if self.train_data != "":
            self.train_data = os.path.join(self.dataset_dir, self.train_data)
            shutil.copytree(self.train_data, self.train_data_dir, dirs_exist_ok=True)
        if self.reg_data != "":
            self.reg_data = os.path.join(self.dataset_dir, self.reg_data)
            shutil.copytree(self.reg_data, self.reg_data_dir, dirs_exist_ok=True)    
        if not os.path.exists(self.accelerate_config):
            write_basic_config(save_location=self.accelerate_config)


        test = os.listdir(self.train_data_dir)
        # @markdown This section will delete unnecessary files and unsupported media such as `.mp4`, `.webm`, and `.gif`.

        supported_types = [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".caption",
            ".combined",
            ".npz",
            ".txt",
            ".json",
        ]

        for item in test:
            file_ext = os.path.splitext(item)[1]
            if file_ext not in supported_types:
                print(f"Deleting file {item} from {self.train_data_dir}")
                os.remove(os.path.join(self.train_data_dir, item))

    def train(self):
        
        lr_scheduler_num_cycles = 0  # @param {'type':'number'}
        lr_scheduler_power = 1 
        lr_warmup_steps = 0 
        noise_offset = 0.0  # @param {type:"number"}

        # sample 
        enable_sample_prompt = True
        mixed_precision = "fp16"  # @param ["no","fp16","bf16"]
        save_precision = "fp16"  # @param ["float", "fp16", "bf16"] 
        save_model_as = "safetensors"  # @param ["ckpt", "safetensors", "diffusers", "diffusers_safetensors"] {allow-input: false}
        max_token_length = 225  # @param {type:"number"}
        gradient_checkpointing = False  # @param {type:"boolean"}
        gradient_accumulation_steps = 1  # @param {type:"number"}
        seed = -1  # @param {type:"number"}



        config = {
            "general": {
                "enable_bucket": True,
                "caption_extension": self.caption_extension,
                "shuffle_caption": True,
                "keep_tokens": self.keep_tokens,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
            },
            "datasets": [
                {
                    "resolution": self.resolution,
                    "min_bucket_reso": 320 if self.resolution > 640 else 256,
                    "max_bucket_reso": 1280 if self.resolution > 640 else 1024,
                    "caption_dropout_rate": self.caption_dropout_rate if self.caption_extension == ".caption" else 0,
                    "caption_tag_dropout_rate": self.caption_dropout_rate if self.caption_extension == ".txt" else 0,
                    "caption_tag_dropout_rate": self.caption_dropout_rate if self.caption_extension == ".combined" else 0,
                    "caption_dropout_every_n_epochs": self.caption_dropout_every_n_epochs,
                    "flip_aug": self.flip_aug,
                    "color_aug": False,
                    "face_crop_aug_range": None,
                    "subsets": [
                        {
                            "image_dir": self.train_data_dir,
                            "class_tokens": self.instance_token,
                            "num_repeats": self.train_repeats,
                        },
                        {
                            "is_reg": True,
                            "image_dir": self.reg_data_dir,
                            "class_tokens": self.class_token,
                            "num_repeats": self.reg_repeats,
                        },
                    ],
                }
            ],
        }

        config_str = toml.dumps(config)

        dataset_config = os.path.join(self.config_dir, "dataset_config.toml")

        for key in config:
            if isinstance(config[key], dict):
                for sub_key in config[key]:
                    if config[key][sub_key] == "":
                        config[key][sub_key] = None
            elif config[key] == "":
                config[key] = None

        config_str = toml.dumps(config)

        with open(dataset_config, "w") as f:
            f.write(config_str)

        # 5.3. Optimizer Config

        optimizer_args = ""  # @param {'type':'string'}
        network_category = "LoRA"  # @param ["LoRA", "LoCon", "LoCon_Lycoris", "LoHa"]

        # @markdown Recommended values:

        # @markdown | network_category | network_dim | network_alpha | conv_dim | conv_alpha |
        # @markdown | :---: | :---: | :---: | :---: | :---: |
        # @markdown | LoRA | 32 | 1 | - | - |
        # @markdown | LoCon | 16 | 8 | 8 | 1 |
        # @markdown | LoHa | 8 | 4 | 4 | 1 |

        # @markdown - Currently, `dropout` and `cp_decomposition` is not available in this notebook.

        # @markdown `conv_dim` and `conv_alpha` are needed to train `LoCon` and `LoHa`, skip it if you train normal `LoRA`. But remember, when in doubt, set `dim = alpha`.
        conv_dim = 32  # @param {'type':'number'}
        conv_alpha = 16  # @param {'type':'number'}
        # @markdown It's recommended to not set `network_dim` and `network_alpha` higher than `64`, especially for LoHa.
        # @markdown But if you want to train with higher dim/alpha so badly, try using higher learning rate. Because the model learning faster in higher dim.

        # @markdown You can specify this field for resume training.
        network_weight = ""  # @param {'type':'string'}
        network_module = (
            "lycoris.kohya"
            if network_category in ["LoHa", "LoCon_Lycoris"]
            else "networks.lora"
        )
        network_args = (
            ""
            if network_category == "LoRA"
            else [
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
            ]
        )

        if network_category == "LoHa":
            network_args.append("algo=loha")
        elif network_category == "LoCon_Lycoris":
            network_args.append("algo=lora")

        # @title ## 5.4. Training Config
        noise_offset = 0.0  # @param {type:"number"}
        save_state = False  # @param {type:"boolean"}

        os.chdir(self.repo_dir)

        config = {
            "model_arguments": {
                "v2": self.v2,
                "v_parameterization": self.v_parameterization if self.v2 and self.v_parameterization else False,
                "pretrained_model_name_or_path": self.sd_path,
                "vae": self.vae_path,
            },
            "additional_network_arguments": {
                "no_metadata": False,
                "unet_lr": float(self.unet_lr) if self.train_unet else None,
                "text_encoder_lr": float(self.text_encoder_lr) if self.train_text_encoder else None,
                "network_weights": network_weight,
                "network_module": network_module,
                "network_dim": self.network_dim,
                "network_alpha": self.network_alpha,
                "network_args": network_args,
                "network_train_unet_only": True if self.train_unet and not self.train_text_encoder else False,
                "network_train_text_encoder_only": True if self.train_text_encoder and not self.train_unet else False,
                "training_comment": None,
            },
            "optimizer_arguments": {
                "optimizer_type": self.optimizer_type,
                "learning_rate": self.unet_lr,
                "max_grad_norm": 1.0,
                "optimizer_args": eval(optimizer_args) if optimizer_args else None,
                "lr_scheduler": self.lr_scheduler,
                "lr_warmup_steps": lr_warmup_steps,
                "lr_scheduler_num_cycles": lr_scheduler_num_cycles if self.lr_scheduler == "cosine_with_restarts" else None,
                "lr_scheduler_power": lr_scheduler_power if self.lr_scheduler == "polynomial" else None,
            },
            "dataset_arguments": {
                "cache_latents": True,
                "debug_dataset": False,
            },
            "training_arguments": {
                "output_dir": self.save_model_dir,
                "output_name": self.project_name,
                "save_precision": save_precision,
                "save_every_n_epochs": self.save_every_n_epochs if self.save_every_n_epochs else None,
                "save_n_epoch_ratio": self.sample_n_epoch_ratio if self.sample_n_epoch_ratio else None,
                "save_last_n_epochs": None,
                "save_state": None,
                "save_last_n_epochs_state": None,
                "resume": None,
                "train_batch_size": self.train_batch_size,
                "max_token_length": 225,
                "mem_eff_attn": False,
                "xformers": False,
                "max_train_epochs": self.num_epochs,
                "max_data_loader_n_workers": 8,
                "persistent_data_loader_workers": True,
                "seed": seed if seed > 0 else None,
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "mixed_precision": mixed_precision,
                "logging_dir": self.logging_dir,
                "log_prefix": self.project_name,
                "noise_offset": noise_offset if noise_offset > 0 else None,
                "lowram": self.lowram,
            },
            "sample_prompt_arguments": {
                "sample_dir": self.sample_dir,
                "sample_every_n_steps": None,
                "sample_every_n_epochs": 1 if enable_sample_prompt else 999999,
                "sample_sampler": self.sampler,
                # "images_per_prompt": images_per_prompt,
            },
            "dreambooth_arguments": {
                "prior_loss_weight": self.prior_loss_weight,
            },
            "saving_arguments": {"save_model_as": save_model_as},
        }

        config_path = os.path.join(self.config_dir, "config_file.toml")
        prompt_path = os.path.join(self.config_dir, "sample_prompt.txt")

        for key in config:
            if isinstance(config[key], dict):
                for sub_key in config[key]:
                    if config[key][sub_key] == "":
                        config[key][sub_key] = None
            elif config[key] == "":
                config[key] = None

        config_str = toml.dumps(config)

        def write_file(filename, contents):
            with open(filename, "w") as f:
                f.write(contents)

        write_file(config_path, config_str)

        final_prompts = []
        for prompt in self.prompts:
            final_prompts.append(
                f"{self.pre}, {prompt} --n {self.negative} --w {self.width} --h {self.height} --l {self.scale} --s {self.steps}" if self.pre else f"{prompt} --n {self.negative} --w {self.width} --h {self.height} --l {self.scale} --s {self.steps}"
            )
        with open(prompt_path, "w") as file:
            # Write each string to the file on a new line
            for string in final_prompts:
                for i in range(self.images_per_prompt):
                    file.write(string + "\n")

        sample_prompt = os.path.join(self.config_dir, "sample_prompt.txt")
        config_file = os.path.join(self.config_dir, "config_file.toml")
        dataset_config = os.path.join(self.config_dir, "dataset_config.toml")
        
        os.chdir(self.repo_dir)
        command = f'''accelerate launch --config_file={self.accelerate_config} --num_cpu_threads_per_process=1 train_network.py --sample_prompts={sample_prompt} --dataset_config={dataset_config} --config_file={config_file}'''
        subprocess.run(command, shell=True, check=True)


