#!/bin/bash
# Setup script for LLM Fine-Tuning Pipeline
# Run this after PyTorch is installed

set -e  # Exit on error

echo "=========================================="
echo "🚀 LLM Fine-Tuning Pipeline Setup"
echo "=========================================="
echo

# Check Python version
echo "📋 Checking Python version..."
python3 --version
echo

# Install core ML dependencies
echo "📦 Installing Transformers, PEFT, BitsAndBytes..."
pip3 install transformers>=4.36.0 peft>=0.7.0 bitsandbytes>=0.41.0 accelerate>=0.25.0

echo
echo "📦 Installing dataset dependencies..."
pip3 install datasets>=2.15.0

echo
echo "📦 Installing ONNX dependencies..."
pip3 install onnx>=1.15.0 onnxruntime-gpu>=1.16.0
pip3 install optimum[onnxruntime-gpu]>=1.16.0

echo
echo "📦 Installing quantization tools..."
pip3 install auto-gptq>=0.6.0 || echo "⚠️ auto-gptq failed (optional)"
pip3 install autoawq>=0.1.8 || echo "⚠️ autoawq failed (optional)"

echo
echo "📦 Installing monitoring tools..."
pip3 install tensorboard>=2.15.0 tqdm>=4.66.0

echo
echo "📦 Installing utilities..."
pip3 install numpy>=1.24.0 pandas>=2.1.0 scikit-learn>=1.3.0

echo
echo "📦 Installing visualization..."
pip3 install matplotlib>=3.8.0 seaborn>=0.13.0

echo
echo "📦 Installing API dependencies..."
pip3 install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 pydantic>=2.5.0

echo
echo "📦 Installing development tools..."
pip3 install jupyter>=1.0.0 ipywidgets>=8.1.0

echo
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Run GPU check: python src/utils/gpu_check.py"
echo "2. Test configuration: python src/utils/config_loader.py"
echo "3. Start training: python src/training/train.py"
echo
