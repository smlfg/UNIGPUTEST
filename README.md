# UNIGPUTEST
Testing the new GPUS

## NVIDIA L40S (48GB VRAM)

Dieses Repository enthält umfassende Informationen und Test-Scripts für die NVIDIA L40S GPU mit 48GB VRAM.

### Inhalt

- **[L40S_CAPABILITIES.md](L40S_CAPABILITIES.md)** - Detaillierte Übersicht über die GPU Capabilities
  - Technische Spezifikationen
  - Anwendungsbereiche (AI/ML, Rendering, Video, etc.)
  - Performance Benchmarks
  - Software Stack Empfehlungen
  - Best Practices

- **[examples/](examples/)** - Praktische Test-Scripts
  - `test_gpu.py` - Grundlegende GPU Tests
  - `llm_inference_test.py` - Large Language Model Tests
  - `stable_diffusion_test.py` - Image Generation Tests
  - Siehe [examples/README.md](examples/README.md) für Details

### Quick Start

```bash
# 1. Repository klonen
git clone <repo-url>
cd UNIGPUTEST

# 2. PyTorch mit CUDA installieren
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Dependencies installieren
pip install -r examples/requirements.txt

# 4. GPU Test ausführen
python examples/test_gpu.py
```

### Hauptanwendungsfälle

Die L40S mit 48GB VRAM eignet sich hervorragend für:

- **Large Language Models**: Llama 2 70B (quantisiert), Mixtral 8x7B, kleinere Modelle in float16
- **Image Generation**: Stable Diffusion XL, multiple SD Instanzen parallel
- **Video Processing**: 8K Encoding/Decoding, Multi-Stream Processing
- **3D Rendering**: Ray-Tracing, CAD/CAM, Virtual Production
- **Scientific Computing**: CFD, Molekulardynamik, Data Science

### Performance Highlights

| Task | Performance |
|------|-------------|
| Llama 2 7B Inference | ~120 tokens/sec |
| SDXL Image Gen (1024x1024) | ~3-5 sec/image |
| Matrix Mult (10k x 10k) | ~180 TFLOPS (TF32) |
| Memory Bandwidth | 864 GB/s |

Mehr Details in [L40S_CAPABILITIES.md](L40S_CAPABILITIES.md)
