# 🚀 LLM Fine-Tuning & Cross-Platform Deployment Pipeline

**Production-ready LLM engineering framework** for efficient fine-tuning (QLoRA) and cross-platform deployment (ONNX).

Optimized for **NVIDIA L40S (48GB VRAM)** with deployment targets including **Snapdragon X Elite NPU**.

---

## 📋 Features

- ✅ **QLoRA Fine-Tuning**: 4-bit quantization with LoRA adapters for memory-efficient training
- ✅ **Multiple Model Support**: Llama 3.2, Llama 3.1, and other HuggingFace models
- ✅ **ONNX Export**: Full pipeline for exporting to ONNX with optimization & quantization
- ✅ **Benchmarking Suite**: Comprehensive performance metrics (latency, throughput, memory)
- ✅ **Production-Ready**: Clean code, modular architecture, extensive documentation
- ✅ **Cross-Platform**: GPU training → ONNX → NPU deployment

---

## 🏗️ Project Structure

```
UNIGPUTEST/
├── src/
│   ├── training/           # Training modules
│   │   ├── train.py       # Main training script
│   │   └── dataset.py     # Dataset loading & preprocessing
│   ├── export/            # Model export
│   │   └── onnx_export.py # ONNX export pipeline
│   ├── evaluation/        # Benchmarking
│   │   └── benchmark.py   # Performance benchmarking
│   └── utils/             # Utilities
│       ├── gpu_check.py   # GPU availability check
│       ├── model_utils.py # Model loading helpers
│       └── config_loader.py # Config management
├── configs/
│   └── training_config.yaml # Training configuration
├── data/                  # Datasets
├── models/                # Saved models
├── checkpoints/           # Training checkpoints
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # Dependencies
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd UNIGPUTEST
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- PyTorch 2.9+ (CUDA 12.8)
- Transformers, PEFT, BitsAndBytes
- ONNX Runtime, Optimum
- TensorBoard/Wandb

### 3. Verify GPU

```bash
python src/utils/gpu_check.py
```

Expected output:
```
✅ CUDA Available: True
✅ GPU: NVIDIA L40S
✅ VRAM: 48.00 GB
```

---

## 🚀 Quick Start

### 1. Fine-Tune a Model

```bash
python src/training/train.py
```

Uses default config (`configs/training_config.yaml`):
- Model: Llama 3.2 3B Instruct
- Dataset: Python code instructions (18k samples)
- Method: QLoRA (4-bit + LoRA adapters)
- Output: `checkpoints/final_model`

### 2. Export to ONNX

```bash
python src/export/onnx_export.py \
  --model-path checkpoints/final_model \
  --output-path models/onnx_model \
  --optimize \
  --quantize
```

### 3. Benchmark Performance

```bash
python src/evaluation/benchmark.py \
  --model-path checkpoints/final_model \
  --device cuda \
  --num-runs 50
```

---

## 📚 Usage Examples

### Training with Custom Config

```python
from src.training.train import main

main(config_path="my_custom_config.yaml")
```

### Loading Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/final_model")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/final_model")

prompt = "Write a Python function to calculate factorial:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0]))
```

### Custom Dataset

```python
from src.training.dataset import CustomDatasetLoader

loader = CustomDatasetLoader(
    data_path="data/my_dataset.json",
    tokenizer=tokenizer,
    max_seq_length=2048,
)

datasets = loader.load()
datasets = loader.preprocess()
```

---

## ⚙️ Configuration

Edit `configs/training_config.yaml` to customize:

### Model Settings

```yaml
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
```

### LoRA Settings

```yaml
lora:
  r: 16                    # LoRA rank
  lora_alpha: 32           # Scaling factor
  lora_dropout: 0.05
  target_modules:          # Modules to adapt
    - "q_proj"
    - "k_proj"
    - "v_proj"
```

### Training Settings

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size: 16
  learning_rate: 2.0e-4
  fp16: false
  bf16: true
```

---

## 📊 Benchmarking Results

Example benchmark on NVIDIA L40S:

| Metric | Llama 3.2 3B (QLoRA) |
|--------|---------------------|
| **Mean Latency** | ~45ms |
| **Throughput** | ~22 tokens/sec |
| **Peak VRAM** | ~6.2 GB |
| **Training Time** | ~2h (18k samples, 3 epochs) |

---

## 🔬 Advanced Usage

### Multi-GPU Training

```yaml
# In config
hardware:
  max_memory:
    0: "46GB"
    1: "46GB"
```

### Custom Prompt Template

```yaml
prompt:
  template: |
    Question: {instruction}
    Answer: {output}
```

### Wandb Logging

```yaml
training:
  report_to: "wandb"
```

Set environment variable:
```bash
export WANDB_API_KEY=your_key
```

---

## 🎯 Target Deployment: Snapdragon X Elite NPU

### ONNX → Qualcomm AI Engine

1. **Export to ONNX** (INT8 quantized)
2. **Convert to Qualcomm DLC** (via Qualcomm AI Engine SDK)
3. **Deploy on NPU** (Windows 11 ARM, WSL)

**Conversion script** (requires Qualcomm SDK):

```bash
# Convert ONNX → DLC
snpe-onnx-to-dlc \
  --input_network models/onnx_model/model_quantized.onnx \
  --output_path models/snapdragon_model.dlc

# Quantize for NPU
snpe-dlc-quantize \
  --input_dlc models/snapdragon_model.dlc \
  --output_dlc models/snapdragon_model_int8.dlc
```

---

## 📈 Performance Optimization Tips

### Memory Optimization

- **Use gradient checkpointing**: Saves VRAM, slightly slower
- **Reduce batch size**: Lower VRAM, longer training
- **Increase LoRA rank**: Better quality, more VRAM

### Speed Optimization

- **Use bf16**: Faster than fp16 on modern GPUs
- **Increase batch size**: Better GPU utilization
- **Use paged_adamw**: More memory-efficient optimizer

### Quality Optimization

- **Increase LoRA rank** (r=32 or r=64)
- **Train longer** (5-10 epochs)
- **Use larger model** (Llama 3.1 8B)

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size in config
per_device_train_batch_size: 2  # Down from 4
gradient_accumulation_steps: 8  # Up from 4
```

### Model Download Issues

```bash
# Set HuggingFace cache
export HF_HOME=/path/to/cache

# Login if using gated models
huggingface-cli login
```

### ONNX Export Fails

```python
# Use manual export fallback
from src.export.onnx_export import ONNXExporter

exporter = ONNXExporter(model_path="...", output_path="...")
exporter.load_model()
exporter._manual_export()  # Fallback method
```

---

## 📖 Documentation

- **Training Guide**: See `docs/training.md` (coming soon)
- **ONNX Export Guide**: See `docs/onnx_export.md` (coming soon)
- **API Reference**: See `docs/api.md` (coming soon)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Transformers** by HuggingFace
- **PEFT** for LoRA implementation
- **BitsAndBytes** for quantization
- **ONNX Runtime** for deployment

---

## 📧 Contact

**Project**: SelfAI - NPU-accelerated LLM
**Author**: [Your Name]
**Institution**: HS Worms - Angewandte Informatik

---

## 🚀 Next Steps

- [ ] Add evaluation metrics (BLEU, ROUGE)
- [ ] Support for more model architectures
- [ ] Docker containerization
- [ ] Web API (FastAPI)
- [ ] Distributed training support
- [ ] Automatic hyperparameter tuning

---

**Built with ❤️ for the ML community**
