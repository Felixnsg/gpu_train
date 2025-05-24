#!/bin/bash
# GPU setup script for cloud training

echo "Setting up Felix Classifier training environment..."

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git htop

# Create virtual environment
python3 -m venv felix_env
source felix_env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Verify GPU access
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

echo "Setup complete!"