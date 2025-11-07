# L40S Test Examples

Praktische Test-Scripts für die NVIDIA L40S GPU.

## Installation

```bash
# CUDA und Treiber müssen bereits installiert sein

# PyTorch mit CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Alle Requirements
pip install -r requirements.txt
```

## Scripts

### 1. test_gpu.py - Grundlegende GPU Tests

Testet CUDA Verfügbarkeit und grundlegende Performance.

```bash
python test_gpu.py
```

**Features:**
- nvidia-smi Output
- CUDA/PyTorch Info
- Memory Status
- Matrix Multiplikation Benchmark
- Inference Performance Test

### 2. llm_inference_test.py - LLM Tests

Testet Large Language Model Inferenz.

```bash
python llm_inference_test.py
```

**Features:**
- GPT-2 Inferenz Test
- Memory Schätzungen für verschiedene Modelle
- 8-bit Quantisierung Beispiele
- Framework Empfehlungen (vLLM, TGI, etc.)

**Unterstützte Modelle:**
- GPT-2 (Small Test)
- Llama 2 7B/13B/70B (mit Code Beispielen)
- Mistral 7B
- Weitere HuggingFace Modelle

### 3. stable_diffusion_test.py - Image Generation

Testet Stable Diffusion Performance.

```bash
python stable_diffusion_test.py
```

**Features:**
- SD 1.5 Benchmark (512x512)
- SDXL Benchmark (1024x1024)
- Batch Size Tests
- Memory Analysen

**Interaktiv:**
```
Wähle Test:
1 - SD 1.5 (schnell)
2 - SDXL (langsam, hohe Qualität)
3 - Batch Size Benchmark
4 - Alle Tests
```

## Erwartete Performance

### LLM Inference
| Modell | Quantisierung | Memory | Tokens/sec |
|--------|---------------|--------|------------|
| Llama 2 7B | float16 | ~14 GB | ~120 |
| Llama 2 13B | float16 | ~26 GB | ~80 |
| Llama 2 70B | int8 | ~40 GB | ~25 |
| Mistral 7B | float16 | ~14 GB | ~140 |

### Image Generation
| Modell | Auflösung | Zeit/Bild | Throughput |
|--------|-----------|-----------|------------|
| SD 1.5 | 512x512 | 1-2s | ~30-60/min |
| SDXL | 1024x1024 | 3-5s | ~12-20/min |
| SD 1.5 Batch 4 | 512x512 | ~1.4s/img | ~40-50/min |

## Optimierungen

### Flash Attention 2
```bash
pip install flash-attn --no-build-isolation
```

Dann in Code:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

### xformers (für Diffusion)
```bash
pip install xformers
```

Automatisch von diffusers genutzt wenn installiert.

### torch.compile()
```python
# 10-20% speedup möglich
model = torch.compile(model)
```

## Production Deployment

### vLLM für LLM Serving
```bash
pip install vllm

# Server starten
vllm serve meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16

# Client
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-2-7b-hf", "prompt": "Hello"}'
```

### ComfyUI für Stable Diffusion
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
python main.py
```

## Monitoring

### GPU Auslastung
```bash
# Echtzeit Monitoring
watch -n 1 nvidia-smi

# Oder mit nvtop (besser)
sudo apt install nvtop
nvtop
```

### Python Monitoring
```python
import torch

# Memory Stats
print(torch.cuda.memory_allocated() / 1024**3)  # GB
print(torch.cuda.memory_reserved() / 1024**3)    # GB
print(torch.cuda.max_memory_allocated() / 1024**3)

# Reset Stats
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
```

## Troubleshooting

### Out of Memory
1. Reduziere Batch Size
2. Nutze Gradient Checkpointing (Training)
3. Nutze Quantisierung (int8/int4)
4. Nutze Flash Attention 2

### Langsame Inferenz
1. Installiere xformers/flash-attn
2. Nutze torch.compile()
3. Nutze TensorRT für Production
4. Prüfe GPU Auslastung (sollte >90% sein)

### CUDA Fehler
```bash
# Prüfe CUDA Version
nvcc --version
nvidia-smi

# Prüfe PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

## Weitere Ressourcen

- [L40S Capabilities](../L40S_CAPABILITIES.md)
- [NVIDIA L40S Docs](https://www.nvidia.com/en-us/data-center/l40s/)
- [PyTorch CUDA Docs](https://pytorch.org/docs/stable/cuda.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Diffusers Docs](https://huggingface.co/docs/diffusers)
