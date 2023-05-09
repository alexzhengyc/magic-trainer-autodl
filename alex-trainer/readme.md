1. 如果是tmp里没有模型，cp autodl-fs/models -r autodl-tmp

2. 如果仍要读取autodl-fs下的sd模型，修改 train_lora/train_dreambooth 的 {sd_path}
    （基本不需要）如果有需要修改blip模型和vae模型的位置，修改 Dreambooth/Lora 中的{self.vae_path/self.blip_path}

3. 运行程序使用automatic1111的venv, python interpreter 输入 stable-diffusion-webui/venv/bin

4. 单次训练文件夹在autodl-tmp/training-outputs 下的{dir_name}, 训练出的模型在 autodl-tmp/models/ 下的lora或者dreambooth 文件夹,可以直接被automatic1111读取

5. 修改configs 中的模型参数，可以设置多个dict来实现多次训练

5. 只运行model = Lora(**config) / model = Dreambooth(**config)， 来生成文件夹

6. 运行model.prepare() 来自动打标签（已有标签注释这行）

7. 运行model.train() 开始训练


export http_proxy=http://10.0.0.7:12798 && export https_proxy=http://10.0.0.7:12798
unset http_proxy && unset https_proxy