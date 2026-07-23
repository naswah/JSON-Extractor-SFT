# QWEN2.5-3B JSON Extractor - Supervised Fine Tuning

A fine-tuned Qwen2.5-3B-Instruct model that converts messy, natural-language input into strict, schema-consistent JSON. Trained with QLoRA via Unsloth on ~1,000 hand-crafted examples (orders, appointments, logs, etc.).

Base model: unsloth/Qwen2.5-3B-Instruct-bnb-4bit\
Method: QLoRA (4-bit base + LoRA adapters, r=16, alpha=32)\
Hardware used: RTX 4070 Ti, 12GB VRAM\
Training environment: WSL2 (Ubuntu) — required for Unsloth/Triton\
Inference environment: Windows (native) — plain transformers + peft

### 🐧 Training in Ubuntu
Unsloth relies on Triton/CUDA kernel compilation that is far more reliable on Linux than native Windows. Training happens in WSL; inference later moves to Windows.

#### 1. Install WSL2+ Ubuntu 
wsl --install -d Ubuntu\
(use python 3.11) for that,
sudo apt update\
sudo apt install -y software-properties-common\
sudo add-apt-repository -y ppa:deadsnakes/ppa\
sudo apt update\
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential

### 2. Setup for project
mkdir -p ~/json-extractor-sft\
cp -r "/mnt/d/JSON Extractor - SFT/"* ~/json-extractor-sft/\
cd ~/json-extractor-sft

### 3. Creating virtual environments and installing dependencies
python3.11 -m venv sft_env\
source sft_env/bin/activate\
pip install --upgrade pip\
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121\
pip install unsloth\
pip install -r requirements.txt

Requirements.txt (WSL)
datasets\
sentencepiece\
protobuf\
huggingface_hub\
ipykernel\
jupyter\
matplotlib\

open vscode and connect it to wsl

### 4. Run the notebook
Run training.py for SFT

## 🪟 Moving trained model from WSL to Windows
rsync -av --progress --exclude='json_env' --exclude='unsloth_compiled_ca*' ./ "/mnt/d/JSON Extractor - SFT/"\
This copies everything (dataset, notebook, trained adapter, loss curve) back to the Windows-visible project folder, while skipping the (large, non-portable) Python virtual environment.

## Windows Inference

### 1. Create a separate env for Windows
python -m venv win_env

### 2. Install dependencies
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121\
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"\
pip install transformers peft accelerate bitsandbytes sentencepiece protobuf

### 3. Run Inference
python interface_windows.py

## 🤖 Model Details
	
Base model- unsloth/Qwen2.5-3B-Instruct-bnb-4bit\
Method- QLoRA (4-bit NF4 base + LoRA adapters)\
LoRA rank / alpha- r=8, alpha=16\
Target modules- q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj\
Trainable params-LoRA adapters only (base model frozen)\
Dataset size- ~1,475 examples, system/user/assistant chat format\
Train/eval split- 80% / 20%\
Epochs- 3\
Patience for earlystopping- 5\
Effective batch size- 16 (batch_size=4 × grad_accum=4)\
Learning rate- 1e-4\
Optimizer- adamw_8bit

### Note
Date: July 23, 2026\
Model overfitting\
Train loss: 0.003, Validation loss: 0.222, model is still memorizing the training data