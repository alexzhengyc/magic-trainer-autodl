#!/usr/bin/env python
# coding: utf-8
# ******************************************************************************************************
## Main Settings

# directory 
dir_name = "chenweiting-768-50-DA-hassan"
project_name = "chenweiting-768-50-DA-hassan"
data_name = "chenweiting/chenweiting-768"

# model
sd_path = "/root/autodl-tmp/webui_models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors"

vae_path = "/root/autodl-tmp/webui_models/VAE/vae-ft-mse-840000-ema-pruned.safetensors"  
blip_path = "/root/autodl-tmp/webui_models/BLIP/model_large_caption.pth"

# dataset
instance_token = "chenweiting" 
class_token = "man"  
add_token_to_caption = True
resolution = 768
flip_aug = True 
data_anotation = "none"  # @param ["none", "waifu", "blip", "combined"]
caption_extension = ".combined"  # @param ["none", ".txt", ".caption", "combined"]

# training 
train_repeats = 2
reg_repeats = 0
num_epochs = 1  # @param {type:"number"}
train_batch_size = 4   # @param {type:"number"}
network_dim = 128 
network_alpha = 128
save_n_epochs_type = "save_every_n_epochs"  # @param ["save_every_n_epochs", "save_n_epoch_ratio"]
save_n_epochs_type_value = 1  # @param {type:"number"}
lr_scheduler = "polynomial"  #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
lowram = False

# sampling
sampler = "k_dpm_2"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
prompts = [
    "1boy, white shirt",
    "1boy, black jacket",
]
scale = 7  # @param {type: "slider", min: 1, max: 40}
sampler = "k_dpm_2"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
steps = 28  # @param {type: "slider", min: 1, max: 100}
# precision = "fp16"  # @param ["fp16", "bf16"] {allow-input: false}
width = 512  # @param {type: "integer"}
height = 512  # @param {type: "integer"}
images_per_prompt = 1  # @param {type: "integer"}

# ******************************************************************************************
# ## Other Settings

root_dir = "/root/alex-sd-training"
output_dir = "/root/autodl-tmp/training-outputs"
save_model_dir = "/root/autodl-fs/webui_models/Lora"
v2 = False 
v_parameterization = False

# dataset
caption_dropout_rate = 0  # @param {type:"slider", min:0, max:1, step:0.05}
caption_dropout_every_n_epochs = 0  
keep_tokens = 0  

# waifu
undesired_tags = ""
general_threshold = 0.3 #@param {type:"slider", min:0, max:1, step:0.05}
character_threshold = 0.5 #@param {type:"slider", min:0, max:1, step:0.05}

# training
optimizer_type = "DAdaptation"  # @param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
train_unet = True  
unet_lr = 1
train_text_encoder = True
text_encoder_lr = 0.5
prior_loss_weight = 1.0
# @markdown Additional arguments for optimizer, e.g: `["decouple=true","weight_decay=0.6"]`
optimizer_args = ""  # @param {'type':'string'}

lr_scheduler_num_cycles = 0  # @param {'type':'number'}
lr_scheduler_power = 1 
lr_warmup_steps = 0 
noise_offset = 0.0  # @param {type:"number"}

# sample 
enable_sample_prompt = True 
pre = "masterpiece, best quality" 
negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"  
clip_skip = None

# 
mixed_precision = "fp16"  # @param ["no","fp16","bf16"]
save_precision = "fp16"  # @param ["float", "fp16", "bf16"] 
save_model_as = "safetensors"  # @param ["ckpt", "pt", "safetensors"] {allow-input: false}
max_token_length = 225  # @param {type:"number"}
gradient_checkpointing = False  # @param {type:"boolean"}
gradient_accumulation_steps = 1  # @param {type:"number"}
seed = -1  # @param {type:"number"}

# **************************************************************************************************

import os
import sys
import shutil
import random
import toml
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import torch
import subprocess
from subprocess import getoutput
from accelerate.utils import write_basic_config

import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"  
os.environ["SAFETENSORS_FAST_GPU"] = "1"

# **************************************************************************************************
repo_dir = os.path.join(root_dir, "kohya-trainer")
training_dir = os.path.join(output_dir, dir_name)
dataset_dir = os.path.join(root_dir, "dataset", data_name)
train_data_dir = os.path.join(training_dir, "train_data")
reg_data_dir = os.path.join(training_dir, "reg_data")
config_dir = os.path.join(training_dir, "config")
accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
tools_dir = os.path.join(repo_dir, "tools")
finetune_dir = os.path.join(repo_dir, "finetune")
sample_dir = os.path.join(training_dir, "sample")
inference_dir = os.path.join(training_dir, "inference")
logging_dir = os.path.join(training_dir, "log")

for dir in [
    training_dir,
    train_data_dir,
    reg_data_dir,
    config_dir,
    output_dir, 
    sample_dir,
    inference_dir
    ]:
    os.makedirs(dir, exist_ok=True)

shutil.copytree(dataset_dir, train_data_dir, dirs_exist_ok=True)
if not os.path.exists(accelerate_config):
    write_basic_config(save_location=accelerate_config)


test = os.listdir(train_data_dir)
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
        print(f"Deleting file {item} from {train_data_dir}")
        os.remove(os.path.join(train_data_dir, item))

# @markdown ### <br> Convert Transparent Images
# @markdown This code will convert your transparent dataset with alpha channel (RGBA) to RGB and give it a white background.

convert = True  # @param {type:"boolean"}
random_color = False  # @param {type:"boolean"}

batch_size = 32

images = [
    image
    for image in os.listdir(train_data_dir)
    if image.endswith(".png") or image.endswith(".webp")
]
background_colors = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def process_image(image_name):
    img = Image.open(f"{train_data_dir}/{image_name}")

    if img.mode in ("RGBA", "LA"):
        if random_color:
            background_color = random.choice(background_colors)
        else:
            background_color = (255, 255, 255)
        bg = Image.new("RGB", img.size, background_color)
        bg.paste(img, mask=img.split()[-1])

        if image_name.endswith(".webp"):
            bg = bg.convert("RGB")
            bg.save(f'{train_data_dir}/{image_name.replace(".webp", ".jpg")}', "JPEG")
            os.remove(f"{train_data_dir}/{image_name}")
            print(
                f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
            )
        else:
            bg.save(f"{train_data_dir}/{image_name}", "PNG")
            print(f" Converted image: {image_name}")
    else:
        if image_name.endswith(".webp"):
            img.save(f'{train_data_dir}/{image_name.replace(".webp", ".jpg")}', "JPEG")
            os.remove(f"{train_data_dir}/{image_name}")
            print(
                f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
            )
        else:
            img.save(f"{train_data_dir}/{image_name}", "PNG")


num_batches = len(images) // batch_size + 1

if convert:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = start + batch_size
            batch = images[start:end]
            executor.map(process_image, batch)

    print("All images have been converted")


# ## Data Annotation
# You can choose to train a model using captions. We're using [BLIP](https://huggingface.co/spaces/Salesforce/BLIP) for image captioning and [Waifu Diffusion 1.4 Tagger](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) for image tagging similar to Danbooru.
# - Use BLIP Captioning for: `General Images`
# - Use Waifu Diffusion 1.4 Tagger V2 for: `Anime and Manga-style Images`
os.chdir(repo_dir)
if data_anotation == "blip" or data_anotation == "combined":

    batch_size = 2 #@param {type:'number'}
    max_data_loader_n_workers = 2 #@param {type:'number'}
    beam_search = True #@param {type:'boolean'}
    min_length = 5 #@param {type:"slider", min:0, max:100, step:5.0}
    max_length = 75 #@param {type:"slider", min:0, max:100, step:5.0}

    command = f'''python make_captions.py "{train_data_dir}" --caption_weights {blip_path} --batch_size {batch_size} {"--beam_search" if beam_search else ""} --min_length {min_length} --max_length {max_length} --caption_extension .caption --max_data_loader_n_workers {max_data_loader_n_workers}'''

    subprocess.run(command, shell=True, check=True)

# 4.2.2. Waifu Diffusion 1.4 Tagger V2

if data_anotation == "waifu" or data_anotation == "combined":

    batch_size = 2 #@param {type:'number'}
    max_data_loader_n_workers = 2 #@param {type:'number'}
    model = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2" #@param ["SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-vit-tagger-v2"]
    #@markdown Use the `recursive` option to process subfolders as well, useful for multi-concept training.
    recursive = False #@param {type:"boolean"} 
    #@markdown Debug while tagging, it will print your image file with general tags and character tags.
    verbose_logging = False #@param {type:"boolean"}
    #@markdown Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.

    config = {
        "_train_data_dir": train_data_dir,
        "batch_size": batch_size,
        "repo_id": model,
        "recursive": recursive,
        "remove_underscore": True,
        "general_threshold": general_threshold,
        "character_threshold": character_threshold,
        "caption_extension": ".txt",
        "max_data_loader_n_workers": max_data_loader_n_workers,
        "debug": verbose_logging,
        "undesired_tags": undesired_tags
    }

    args = ""
    for k, v in config.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    final_args = f"python tag_images_by_wd14_tagger.py {args}"

    subprocess.run(final_args, shell=True, check=True)


# ### Combine BLIP and Waifu

if data_anotation == "combined":
    def read_file_content(file_path):
        with open(file_path, "r") as file:
            content = file.read()
        return content

    def remove_redundant_words(content1, content2):
        return content1.rstrip('\n') + ', ' + content2

    def write_file_content(file_path, content):
        with open(file_path, "w") as file:
            file.write(content)

    def combine():
        directory = train_data_dir
        extension1 = ".caption"
        extension2 = ".txt"
        output_extension = ".combined"

        for file in os.listdir(directory):
            if file.endswith(extension1):
                filename = os.path.splitext(file)[0]
                file1 = os.path.join(directory, filename + extension1)
                file2 = os.path.join(directory, filename + extension2)
                output_file = os.path.join(directory, filename + output_extension)

                if os.path.exists(file2):
                    content1 = read_file_content(file1)
                    content2 = read_file_content(file2)

                    combined_content = remove_redundant_words(content1, content2)

                    write_file_content(output_file, combined_content)

    combine()


# ## Training Model


# @title ## 5.2. Dataset Config

if add_token_to_caption and keep_tokens < 2:
    keep_tokens = 1

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

def add_tag(filename, tag):
    contents = read_file(filename)
    # move the "cat" or "dog" to the beginning of the contents
    if "cat" in contents:
        contents = contents.replace("cat, ", "")
        contents = contents.replace(", cat", "")
        contents = "cat, " + contents
    if "dog" in contents:
        contents = contents.replace("dog, ", "")
        contents = contents.replace(", dog", "")
        contents = "dog, " + contents

    # add the tag
    tag = ", ".join(tag.split())
    tag = tag.replace("_", " ")
    if tag in contents:
        return
    contents = tag + ", " + contents
    write_file(filename, contents)

def delete_tag(filename, tag):
    contents = read_file(filename)
    tag = ", ".join(tag.split())
    tag = tag.replace("_", " ")
    if tag not in contents:
        return
    contents = "".join([s.strip(", ") for s in contents.split(tag)])
    write_file(filename, contents)

if caption_extension != "none":

    tag = f"{instance_token}_{class_token}" if 'class_token' in globals() else instance_token
    for filename in os.listdir(train_data_dir):
        if filename.endswith(caption_extension):
            file_path = os.path.join(train_data_dir, filename)

            if add_token_to_caption:
                add_tag(file_path, tag)
            else:
                delete_tag(file_path, tag)

config = {
    "general": {
        "enable_bucket": True,
        "caption_extension": caption_extension,
        "shuffle_caption": True,
        "keep_tokens": keep_tokens,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
    },
    "datasets": [
        {
            "resolution": resolution,
            "min_bucket_reso": 320 if resolution > 640 else 256,
            "max_bucket_reso": 1280 if resolution > 640 else 1024,
            "caption_dropout_rate": caption_dropout_rate if caption_extension == ".caption" else 0,
            "caption_tag_dropout_rate": caption_dropout_rate if caption_extension == ".txt" else 0,
            "caption_tag_dropout_rate": caption_dropout_rate if caption_extension == ".combined" else 0,
            "caption_dropout_every_n_epochs": caption_dropout_every_n_epochs,
            "flip_aug": flip_aug,
            "color_aug": False,
            "face_crop_aug_range": None,
            "subsets": [
                {
                    "image_dir": train_data_dir,
                    "class_tokens": f"{instance_token} {class_token}" if 'class_token' in globals() else instance_token,
                    "num_repeats": train_repeats,
                },
                {
                    "is_reg": True,
                    "image_dir": reg_data_dir,
                    "class_tokens": class_token if 'class_token' in globals() else None,
                    "num_repeats": reg_repeats,
                },
            ],
        }
    ],
}

config_str = toml.dumps(config)

dataset_config = os.path.join(config_dir, "dataset_config.toml")

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

print(config_str)

# 5.3. LoRA and Optimizer Config

# @markdown ### LoRA Config:
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
network_module = "lycoris.kohya" if network_category in ["LoHa", "LoCon_Lycoris"] else "networks.lora"
network_args = "" if network_category == "LoRA" else [
    f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}",
    ]

if network_category == "LoHa":
  network_args.append("algo=loha")
elif network_category == "LoCon_Lycoris":
  network_args.append("algo=lora")

print("- LoRA Config:")
print(f"  - Additional network category: {network_category}")
print(f"  - Loading network module: {network_module}")
if not network_category == "LoRA":
  print(f"  - network args: {network_args}")
print(f"  - {network_module} linear_dim set to: {network_dim}")
print(f"  - {network_module} linear_alpha set to: {network_alpha}")
if not network_category == "LoRA":
  print(f"  - {network_module} conv_dim set to: {conv_dim}")
  print(f"  - {network_module} conv_alpha set to: {conv_alpha}")

if not network_weight:
    print("  - No LoRA weight loaded.")
else:
    if os.path.exists(network_weight):
        print(f"  - Loading LoRA weight: {network_weight}")
    else:
        print(f"  - {network_weight} does not exist.")
        network_weight = ""

print("- Optimizer Config:")
print(f"  - Using {optimizer_type} as Optimizer")
if optimizer_args:
    print(f"  - Optimizer Args: {optimizer_args}")
if train_unet and train_text_encoder:
    print("  - Train UNet and Text Encoder")
    print(f"    - UNet learning rate: {unet_lr}")
    print(f"    - Text encoder learning rate: {text_encoder_lr}")
if train_unet and not train_text_encoder:
    print("  - Train UNet only")
    print(f"    - UNet learning rate: {unet_lr}")
if train_text_encoder and not train_unet:
    print("  - Train Text Encoder only")
    print(f"    - Text encoder learning rate: {text_encoder_lr}")
print(f"  - Learning rate warmup steps: {lr_warmup_steps}")
print(f"  - Learning rate Scheduler: {lr_scheduler}")
if lr_scheduler == "cosine_with_restarts":
    print(f"  - lr_scheduler_num_cycles: {lr_scheduler_num_cycles}")
elif lr_scheduler == "polynomial":
    print(f"  - lr_scheduler_power: {lr_scheduler_power}")


# @title ## 5.4. Training Config


os.chdir(repo_dir)

config = {
    "model_arguments": {
        "v2": v2,
        "v_parameterization": v_parameterization
        if v2 and v_parameterization
        else False,
        "pretrained_model_name_or_path": sd_path,
        "vae": vae_path,
    },
    "additional_network_arguments": {
        "no_metadata": False,
        "unet_lr": float(unet_lr) if train_unet else None,
        "text_encoder_lr": float(text_encoder_lr) if train_text_encoder else None,
        "network_weights": network_weight,
        "network_module": network_module,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_args": network_args,
        "network_train_unet_only": True if train_unet and not train_text_encoder else False,
        "network_train_text_encoder_only": True if train_text_encoder and not train_unet else False,
        "training_comment": None,
    },
    "optimizer_arguments": {
        "optimizer_type": optimizer_type,
        "learning_rate": unet_lr,
        "max_grad_norm": 1.0,
        "optimizer_args": eval(optimizer_args) if optimizer_args else None,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": lr_warmup_steps,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
    },
    "dataset_arguments": {
        "cache_latents": True,
        "debug_dataset": False,
    },
    "training_arguments": {
        "output_dir": save_model_dir,
        "output_name": project_name,
        "save_precision": save_precision,
        "save_every_n_epochs": save_n_epochs_type_value if save_n_epochs_type == "save_every_n_epochs" else None,
        "save_n_epoch_ratio": save_n_epochs_type_value if save_n_epochs_type == "save_n_epoch_ratio" else None,
        "save_last_n_epochs": None,
        "save_state": None,
        "save_last_n_epochs_state": None,
        "resume": None,
        "train_batch_size": train_batch_size,
        "max_token_length": 225,
        "mem_eff_attn": False,
        "xformers": False,
        "max_train_epochs": num_epochs,
        "max_data_loader_n_workers": 8,
        "persistent_data_loader_workers": True,
        "seed": seed if seed > 0 else None,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "logging_dir": logging_dir,
        "log_prefix": project_name,
        "noise_offset": noise_offset if noise_offset > 0 else None,
        "lowram": lowram,
        "clip_skip": clip_skip,
    },
    "sample_prompt_arguments": {
        "sample_every_n_steps": None,
        "sample_every_n_epochs": 1 if enable_sample_prompt else 999999,
        "sample_sampler": sampler,
        # "images_per_prompt": images_per_prompt,
    },
    "dreambooth_arguments": {
        "prior_loss_weight": prior_loss_weight,
    },
    "saving_arguments": {"save_model_as": save_model_as},
}

config_path = os.path.join(config_dir, "config_file.toml")
prompt_path = os.path.join(config_dir, "sample_prompt.txt")

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
for prompt in prompts:
    final_prompts.append(
    f"{instance_token}, {pre}, {prompt} --n {negative} --w {width} --h {height} --l {scale} --s {steps}"
    if add_token_to_caption
    else f"{pre}, {prompt} --n {negative} --w {width} --h {height} --l {scale} --s {steps}"
    )
with open(prompt_path, 'w') as file:
    # Write each string to the file on a new line
    for string in final_prompts:
        for i in range(images_per_prompt):
            file.write(string + '\n')
    
print(config_str)

sample_prompt = os.path.join(config_dir, "sample_prompt.txt")
config_file = os.path.join(config_dir, "config_file.toml")
dataset_config = os.path.join(config_dir, "dataset_config.toml")
 
os.chdir(repo_dir)
command = f'''accelerate launch --config_file={accelerate_config} --num_cpu_threads_per_process=1 train_network.py --sample_prompts={sample_prompt} --dataset_config={dataset_config} --config_file={config_file}'''

subprocess.run(command, shell=True, check=True)