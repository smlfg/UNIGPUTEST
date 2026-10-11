#!/bin/bash
# Phase 1: Install PyTorch and Core Dependencies
# For NVIDIA L40S GPU Testing

set -e

echo "========================================"
echo "Phase 1: PyTorch Installation"
echo "========================================"
echo ""

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio

echo ""
echo "Installing GPU monitoring tools..."
pip install nvidia-ml-py3 gpustat

echo ""
echo "Installing scientific computing essentials..."
pip install numpy pandas matplotlib seaborn

echo ""
echo "Installing development tools..."
pip install ipython jupyter tqdm

echo ""
echo "========================================"
echo "Phase 1 Complete!"
echo "========================================"
echo ""
echo "Next step: Run python gpu_info.py to verify GPU setup"
