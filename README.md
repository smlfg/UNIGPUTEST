# NVIDIA L40S GPU Testing Suite

Comprehensive testing, benchmarking, and capabilities documentation for NVIDIA L40S GPUs (48GB VRAM).

## Overview

This repository provides a complete framework for:
- **GPU validation and benchmarking** - Performance testing for PyTorch, TensorFlow, and CUDA
- **AI/ML capabilities** - LLM inference, image generation, and model deployment guides
- **Production workloads** - Best practices for real-world applications

### GPU Specifications
- **Model**: NVIDIA L40S
- **Memory**: 48GB GDDR6 with ECC
- **Architecture**: Ada Lovelace
- **CUDA Cores**: 18,176
- **Tensor Cores**: 568 (4th Gen)
- **RT Cores**: 142 (3rd Gen)
- **FP32 Performance**: 91.6 TFLOPS
- **FP16 Performance**: 183.2 TFLOPS (with Tensor Cores: 733 TFLOPS)
- **Memory Bandwidth**: 864 GB/s

## Contents

### 📊 Benchmark & Testing Suite
- **[gpu_info.py](gpu_info.py)** - GPU verification and information
- **[gpu_benchmark.py](gpu_benchmark.py)** - Comprehensive performance benchmarks
- **[test_pytorch.py](test_pytorch.py)** - PyTorch-specific tests
- **[test_tensorflow.py](test_tensorflow.py)** - TensorFlow-specific tests

### 🚀 AI/ML Capabilities & Examples
- **[L40S_CAPABILITIES.md](L40S_CAPABILITIES.md)** - Complete capabilities overview
  - Use cases: LLMs, Image Generation, Video, 3D Rendering, Scientific Computing
  - Model recommendations and memory estimates
  - Software stack and framework guides
  - Production deployment strategies

- **[examples/](examples/)** - Practical test scripts
  - `test_gpu.py` - Basic GPU and CUDA tests
  - `llm_inference_test.py` - LLM inference examples
  - `stable_diffusion_test.py` - Image generation tests
  - See [examples/README.md](examples/README.md) for details

## Quick Start

### Prerequisites
- NVIDIA L40S GPU
- NVIDIA Driver >= 525.xx
- CUDA Toolkit >= 12.0
- Python >= 3.8

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd UNIGPUTEST

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. For AI/ML examples, install additional packages
pip install -r examples/requirements.txt
```

### Verify Your Setup

```bash
# Check GPU detection
python gpu_info.py

# Run basic benchmarks
python gpu_benchmark.py

# Test PyTorch
python test_pytorch.py

# Test LLM capabilities (requires transformers)
python examples/llm_inference_test.py
```

## Usage Guide

### 1. GPU Verification & Benchmarking

#### Check GPU Status
```bash
python gpu_info.py
```
Displays:
- NVIDIA driver and GPU information
- CUDA toolkit installation
- PyTorch and TensorFlow GPU access
- Detailed GPU specifications

#### Run Performance Benchmarks
```bash
# Run all benchmarks
python gpu_benchmark.py

# Framework-specific
python gpu_benchmark.py --framework pytorch
python gpu_benchmark.py --framework tensorflow
```

Benchmarks include:
- Matrix multiplication (FP32, FP16, BF16)
- Memory bandwidth tests
- Convolution operations
- Element-wise operations
- Memory management

### 2. Framework Testing

#### PyTorch
```bash
python test_pytorch.py
```
Tests: Tensor operations, mixed precision, CNN training, memory management

#### TensorFlow
```bash
python test_tensorflow.py
```
Tests: Tensor operations, mixed precision, training, tf.data pipeline

### 3. AI/ML Workloads

#### LLM Inference
```bash
python examples/llm_inference_test.py
```
- Tests GPT-2 and provides guidance for larger models
- Shows memory estimates for Llama 2, Mistral, etc.
- Demonstrates quantization techniques

#### Image Generation
```bash
python examples/stable_diffusion_test.py
```
- Stable Diffusion 1.5 and XL benchmarks
- Batch size optimization
- Performance metrics

## Performance Expectations

### Core Compute
| Operation | Performance |
|-----------|-------------|
| Matrix Mult 4096x4096 (FP32) | 60-80 TFLOPS, ~5-10 ms |
| Memory Bandwidth | ~800-900 GB/s |
| CNN Training (batch 32) | 300-500 images/sec |
| Mixed Precision Speedup | 2-3x (FP16/BF16 vs FP32) |

### AI/ML Workloads
| Task | Performance |
|------|-------------|
| Llama 2 7B Inference | ~120 tokens/sec |
| Llama 2 13B Inference | ~80 tokens/sec |
| Stable Diffusion 1.5 (512x512) | 1-2 sec/image |
| SDXL (1024x1024) | 3-5 sec/image |

## Key Capabilities

### Large Language Models
- **Inference**: Llama 2 70B (int8), Mixtral 8x7B, Falcon 40B (quantized)
- **Fine-tuning**: Llama 2 7B/13B (full), larger models with LoRA/QLoRA
- **Serving**: Production API with vLLM, TGI

### Generative AI
- **Image**: Stable Diffusion XL, ControlNet, multiple models parallel
- **Video**: AnimateDiff, Stable Video Diffusion
- **Audio**: Whisper, Bark, MusicGen

### Other Workloads
- **3D Rendering**: Ray-tracing, Blender, Unreal Engine
- **Video**: 8K encoding/decoding, multi-stream processing
- **Scientific**: CFD, molecular dynamics, data science

See [L40S_CAPABILITIES.md](L40S_CAPABILITIES.md) for comprehensive details.

## Optimization Tips

### PyTorch
- Enable `torch.backends.cudnn.benchmark = True` for fixed input sizes
- Use mixed precision with `torch.cuda.amp`
- Use `torch.compile()` for PyTorch 2.0+
- Install flash-attention-2 for transformers

### TensorFlow
- Enable XLA: `tf.config.optimizer.set_jit(True)`
- Mixed precision: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- Memory growth: `tf.config.experimental.set_memory_growth(gpu, True)`

### General
- Use batch sizes that are multiples of 8 for tensor cores
- Prefer FP16/BF16 for training when possible
- For LLMs: use int8/int4 quantization for larger models
- For image generation: use xformers or torch.compile()

## Monitoring

```bash
# Continuous GPU monitoring
nvidia-smi -l 1

# More readable (install: pip install gpustat)
gpustat -i 1

# Detailed watch
watch -n 1 nvidia-smi
```

## Troubleshooting

### CUDA Not Available
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
nvcc --version

# Set CUDA path if needed
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Out of Memory
- L40S has 48GB VRAM - most workloads fit comfortably
- Reduce batch size if needed
- Use gradient checkpointing for training
- Use quantization (int8/int4) for large models
- Clear cache: `torch.cuda.empty_cache()`

### Driver/CUDA Mismatch
```bash
# Check versions
nvidia-smi  # Shows driver CUDA version
nvcc --version  # Shows toolkit version

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Repository Structure

```
UNIGPUTEST/
├── README.md                      # This file
├── L40S_CAPABILITIES.md           # Comprehensive capabilities guide
├── requirements.txt               # Core dependencies
│
├── gpu_info.py                    # GPU verification script
├── gpu_benchmark.py               # Performance benchmarks
├── test_pytorch.py                # PyTorch tests
├── test_tensorflow.py             # TensorFlow tests
│
└── examples/                      # AI/ML examples
    ├── README.md                  # Examples documentation
    ├── requirements.txt           # AI/ML dependencies
    ├── test_gpu.py               # Basic GPU tests
    ├── llm_inference_test.py     # LLM examples
    └── stable_diffusion_test.py  # Image generation
```

## Additional Resources

- [NVIDIA L40S Documentation](https://www.nvidia.com/en-us/data-center/l40s/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)

## Contributing

Contributions welcome! Feel free to add tests, benchmarks, examples, or documentation improvements.

## License

MIT License
