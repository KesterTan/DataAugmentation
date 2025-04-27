# Upgrade pip
pip install --upgrade pip

# Uninstall old CPU-only versions of PyTorch, torchvision, torchaudio
pip uninstall torch torchvision torchaudio bitsandbytes -y

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reinstall bitsandbytes (correct CUDA-enabled version)
pip install bitsandbytes --no-cache-dir

# (Optional) Also install or upgrade Huggingface libraries if needed
pip install transformers datasets accelerate peft --upgrade

