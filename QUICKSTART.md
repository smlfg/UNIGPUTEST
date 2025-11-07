# 🚀 QUICKSTART - NVIDIA L40S LLM Development

Schritt-für-Schritt Anleitung für LLM Development auf der NVIDIA L40S GPU.

## 📋 Übersicht

Diese Anleitung führt dich durch:
1. ✅ PyTorch & GPU Setup
2. 🤖 LLM Loading & Testing
3. ⚡ Quantization Experiments
4. 🎓 Fine-Tuning Setup
5. 📊 Performance Optimization

---

## Phase 1: PyTorch Installation & GPU Verification

### Schritt 1.1: PyTorch installieren

```bash
cd /home/user/UNIGPUTEST

# Aktiviere conda environment
conda activate gpu-test

# Installiere PyTorch & Dependencies
bash install_phase1_pytorch.sh
```

**Dauer:** ~5-10 Minuten (3GB Download)

### Schritt 1.2: GPU verifizieren

```bash
# GPU Info Check
python gpu_info.py

# Sollte anzeigen:
# ✓ NVIDIA Driver: Installed
# ✓ CUDA Available: True
# ✓ PyTorch: <version>
# ✓ GPU: NVIDIA L40S
```

### Schritt 1.3: Benchmarks laufen lassen

```bash
# Vollständige Benchmarks
python gpu_benchmark.py

# Nur PyTorch Benchmarks
python gpu_benchmark.py --framework pytorch

# PyTorch Tests
python test_pytorch.py
```

**Erwartete Performance:**
- Matrix Multiplication (4096x4096): 60-80 TFLOPS
- FP16 vs FP32: 2-3x schneller
- Memory Bandwidth: ~800-900 GB/s

---

## Phase 2: Transformers & LLM Tools

### Schritt 2.1: Dependencies installieren

```bash
# Installiere Transformers, PEFT, etc.
bash install_phase2_transformers.sh
```

**Dauer:** ~3-5 Minuten

### Schritt 2.2: Test LLM Loading

```bash
# Test mit GPT-2 (klein, schnell)
python test_llm_loading.py
```

Dies testet:
- ✅ Model Loading (FP16, 8-bit, 4-bit)
- ✅ GPU Memory Usage
- ✅ Inference Speed
- ✅ Text Generation

**Erwartete Ausgabe:**
```
Model           Quant    Load Time    Memory       Tokens/s     Status
----------------------------------------------------------------------
gpt2            FP16     2.50         0.48         125.00       ✓
gpt2            8bit     3.20         0.25         120.00       ✓
gpt2            4bit     3.80         0.13         115.00       ✓
```

---

## Phase 3: Größere Modelle testen

### Schritt 3.1: Llama 2 7B laden (optional - braucht HF Token)

```python
# In Python oder Jupyter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-2-7b-hf"  # Braucht HF Access Token

# FP16 (~14GB)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 8-bit (~7GB) - Empfohlen!
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit (~3.5GB) - Noch besser!
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
```

### Schritt 3.2: Mistral 7B (keine Token nötig!)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"

# 4-bit Loading (empfohlen)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test inference
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## Phase 4: Performance Monitoring

### GPU Monitoring während der Arbeit

```bash
# Terminal 1: Continuous monitoring
watch -n 1 nvidia-smi

# Terminal 2: Better formatted output
gpustat -i 1

# Terminal 3: Your work
python test_llm_loading.py
```

### Memory Check

```python
import torch

# Check available memory
total = torch.cuda.get_device_properties(0).total_memory / 1e9
allocated = torch.cuda.memory_allocated(0) / 1e9
free = total - allocated

print(f"Total: {total:.1f} GB")
print(f"Allocated: {allocated:.1f} GB")
print(f"Free: {free:.1f} GB")

# Clear cache wenn nötig
torch.cuda.empty_cache()
```

---

## Phase 5: Next Steps

### Fine-Tuning vorbereiten

```bash
# Install zusätzliche Tools
pip install wandb  # Für Training Monitoring
pip install peft   # Für LoRA/QLoRA (schon installiert)
```

### Jupyter Notebook starten

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Weitere Modelle zum Testen

**Kleine Modelle (gut für Tests):**
- `gpt2` (~500MB)
- `gpt2-medium` (~1.5GB)
- `facebook/opt-1.3b` (~2.6GB)

**Mittlere Modelle:**
- `mistralai/Mistral-7B-v0.1` (~14GB FP16, ~3.5GB 4-bit)
- `meta-llama/Llama-2-7b-hf` (~14GB FP16, ~3.5GB 4-bit)
- `tiiuae/falcon-7b` (~14GB FP16)

**Große Modelle (L40S kann das!):**
- `meta-llama/Llama-2-13b-hf` (~26GB FP16, ~6.5GB 4-bit)
- `mistralai/Mixtral-8x7B-v0.1` (~90GB FP16, ~23GB 4-bit)

---

## 🆘 Troubleshooting

### CUDA Out of Memory
```python
# Clear GPU memory
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

### Slow Loading
- Nutze `device_map="auto"` für automatisches GPU mapping
- Nutze 4-bit quantization für große Modelle
- Cache models lokal: `export HF_HOME=/path/to/cache`

### Import Errors
```bash
# Reinstall package
pip uninstall <package> -y
pip install <package> --no-cache-dir
```

---

## 📚 Nützliche Links

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [bitsandbytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [NVIDIA L40S Specs](https://www.nvidia.com/en-us/data-center/l40s/)

---

## 🎯 Quick Reference Commands

```bash
# GPU Check
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Installation
bash install_phase1_pytorch.sh
bash install_phase2_transformers.sh

# Testing
python gpu_info.py
python gpu_benchmark.py
python test_llm_loading.py

# Monitoring
gpustat -i 1
watch -n 1 nvidia-smi
```

---

**Viel Erfolg! 🚀**
