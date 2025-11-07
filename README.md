# NVIDIA L40S GPU Testing Suite

Comprehensive testing and benchmarking suite for NVIDIA L40S GPUs (48GB VRAM) with support for PyTorch, TensorFlow, and CUDA operations.

## Overview

This repository provides a complete testing framework for validating GPU setup, running performance benchmarks, and testing machine learning frameworks on NVIDIA L40S GPUs.

### GPU Specifications
- **Model**: NVIDIA L40S
- **Memory**: 48GB GDDR6
- **Architecture**: Ada Lovelace
- **CUDA Cores**: 18,176
- **Tensor Cores**: 568 (4th Gen)
- **FP32 Performance**: 91.6 TFLOPS
- **FP16 Performance**: 183.2 TFLOPS (with Tensor Cores: 733 TFLOPS)

## Prerequisites

### System Requirements
- NVIDIA L40S GPU
- NVIDIA Driver >= 525.xx
- CUDA Toolkit >= 12.0
- Python >= 3.8

### Check Your Setup
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Verify GPU is detected
lspci | grep -i nvidia
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd UNIGPUTEST
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n gpu-test python=3.10
conda activate gpu-test
```

### 3. Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually:
# PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow
pip install tensorflow[and-cuda]

# Additional GPU tools
pip install nvidia-ml-py3 gpustat
```

## Usage

### 1. GPU Information and Verification
Check GPU status, drivers, CUDA installation, and framework availability:

```bash
python gpu_info.py
```

This script will:
- Display NVIDIA driver and GPU information
- Check CUDA toolkit installation
- Verify PyTorch and TensorFlow GPU access
- Test basic tensor operations
- Show detailed GPU specifications

### 2. GPU Benchmarks
Run comprehensive performance benchmarks:

```bash
# Run all benchmarks
python gpu_benchmark.py

# Run PyTorch benchmarks only
python gpu_benchmark.py --framework pytorch

# Run TensorFlow benchmarks only
python gpu_benchmark.py --framework tensorflow
```

Benchmarks include:
- **Matrix Multiplication**: FP32, FP16, BF16 precision tests
- **Memory Bandwidth**: Large tensor allocation and transfer
- **Convolution Operations**: CNN-like workloads
- **Element-wise Operations**: Add, multiply, exp, sin, sqrt
- **Memory Management**: Allocation and deallocation tests

### 3. Framework-Specific Tests

#### PyTorch Tests
```bash
python test_pytorch.py
```

Tests:
- Basic tensor operations
- Mixed precision (FP16/BF16) training
- Neural network training (CNN)
- Memory management
- Multi-GPU support (if available)

#### TensorFlow Tests
```bash
python test_tensorflow.py
```

Tests:
- Basic tensor operations
- Mixed precision training
- Neural network training
- tf.data pipeline
- Memory information

## Expected Performance

### Matrix Multiplication (FP32)
- 4096x4096: ~5-10 ms, 60-80 TFLOPS

### Mixed Precision Speedup
- FP16 vs FP32: 2-3x faster
- BF16 vs FP32: 2-3x faster

### Memory Bandwidth
- ~800-900 GB/s for large transfers

### CNN Training
- ResNet-like layer (batch 32): 300-500 images/sec

## Troubleshooting

### CUDA Not Available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA path
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Set CUDA path if needed
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Out of Memory Errors
- The L40S has 48GB VRAM, which should handle most workloads
- Reduce batch sizes or model sizes if needed
- Clear GPU cache: `torch.cuda.empty_cache()` (PyTorch)

### Driver/CUDA Version Mismatch
```bash
# Check driver CUDA version
nvidia-smi

# Check toolkit version
nvcc --version

# Ensure PyTorch/TensorFlow matches your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
```

## File Structure

```
UNIGPUTEST/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── gpu_info.py              # GPU information and verification script
├── gpu_benchmark.py         # Comprehensive benchmark suite
├── test_pytorch.py          # PyTorch-specific tests
└── test_tensorflow.py       # TensorFlow-specific tests
```

## Performance Tips

### For PyTorch
- Enable `torch.backends.cudnn.benchmark = True` for fixed input sizes
- Use mixed precision training with `torch.cuda.amp`
- Use `torch.compile()` for PyTorch 2.0+

### For TensorFlow
- Enable XLA compilation: `tf.config.optimizer.set_jit(True)`
- Use mixed precision policy: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- Enable memory growth: `tf.config.experimental.set_memory_growth(gpu, True)`

### General
- Use batch sizes that are multiples of 8 for tensor cores
- Prefer FP16/BF16 for training when possible
- Monitor GPU utilization with `nvidia-smi -l 1` or `gpustat -i 1`

## Monitoring GPU Usage

```bash
# Continuous monitoring
nvidia-smi -l 1

# With gpustat (more readable)
gpustat -i 1

# Watch memory and utilization
watch -n 1 nvidia-smi
```

## Additional Resources

- [NVIDIA L40S Documentation](https://www.nvidia.com/en-us/data-center/l40s/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)

## Contributing

Feel free to add more tests, benchmarks, or improvements!

## License

MIT License
