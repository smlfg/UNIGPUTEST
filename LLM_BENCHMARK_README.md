# Advanced LLM Benchmark Suite for NVIDIA L40S

Umfassendes Benchmark-System für Large Language Models mit erweiterten Metriken, Monitoring und automatischer Berichterstellung.

## 🚀 Features

### 1. **Unterstützte Modelle**
- ✅ **Falcon**: 7B, 13B
- ✅ **Mistral**: 7B
- ✅ **Llama 2**: 7B, 13B
- ✅ **CodeLlama**: 7B, 34B
- ✅ **GPT-2**: Small, Medium, Large, XL

### 2. **Quantisierungen**
- **FP32**: Full Precision (32-bit)
- **FP16**: Half Precision (16-bit)
- **BF16**: Brain Float 16
- **INT8**: 8-bit Quantisierung (bitsandbytes)
- **INT4**: 4-bit Quantisierung (NF4)
- **Dynamic**: PyTorch Dynamic Quantization

### 3. **Erweiterte Metriken**
- ⚡ **Performance**: Tokens/s, Latency, Throughput
- 💾 **Memory**: Allocated, Reserved, Peak, Fragmentation
- 🌡️ **Thermal**: GPU Temperature (Avg/Peak)
- ⚡ **Power**: Power Consumption in Watt
- 📊 **GPU**: Utilization, Clock Speeds
- 🔄 **Cache**: Cache Hit Rates

### 4. **Batch Processing**
- Automatisches Testing verschiedener Batch-Größen (1, 2, 4, 8, 16, 32)
- Scaling-Effizienz-Analyse
- Durchsatz-Optimierung

### 5. **Automatisierte Reports**
- 📄 **JSON**: Strukturierte Daten für weitere Verarbeitung
- 📊 **CSV**: Excel-kompatibel für Analysen
- 📝 **Markdown**: Lesbare Berichte mit Tabellen
- 📈 **Visualisierungen**: Plots und Diagramme
- 🔄 **Session-Vergleiche**: Trend-Analysen über Zeit

## 📦 Installation

### Voraussetzungen
- NVIDIA L40S GPU (48GB VRAM)
- CUDA 11.8 oder höher
- Python 3.8+

### Setup

```bash
# Repository klonen
git clone <your-repo-url>
cd UNIGPUTEST

# Virtual Environment erstellen
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt

# PyTorch mit CUDA (falls nicht in requirements.txt)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: HuggingFace Login für gated models (Llama 2)
huggingface-cli login
```

## 🎯 Verwendung

### Quick Start

```bash
# Schneller Test mit GPT-2
python llm_benchmark.py --quick

# Einzelnes Modell benchmarken
python llm_benchmark.py --model gpt2 --quantization fp16 --batch-sizes 1 4 8

# Mistral 7B mit verschiedenen Quantisierungen
python llm_benchmark.py --model mistralai/Mistral-7B-v0.1 \
    --quantization int8 --batch-sizes 1 2 4 8 16

# Llama 2 13B mit INT4 Quantisierung
python llm_benchmark.py --model meta-llama/Llama-2-13b-hf \
    --quantization int4 --batch-sizes 1 2 4
```

### Vollständige Test-Suite

```bash
# Alle Modelle mit allen Quantisierungen (dauert mehrere Stunden!)
python llm_benchmark.py --config model_config.yaml

# Nur 7B Modelle
python llm_benchmark.py \
    --model mistralai/Mistral-7B-v0.1 \
    --model meta-llama/Llama-2-7b-hf \
    --model codellama/CodeLlama-7b-hf \
    --quantization fp16 int8
```

### GPU Monitoring

```bash
# Echtzeit GPU Monitoring mit curses UI
python gpu_monitor.py

# Einfacher Output (für Logging)
python gpu_monitor.py --simple

# Monitoring für 300 Sekunden
python gpu_monitor.py --duration 300

# Schnelleres Update (10x pro Sekunde)
python gpu_monitor.py --interval 0.1
```

### Report Generierung

```bash
# Alle Reports generieren
python report_generator.py --results-dir benchmark_results --plots

# Nur Markdown Report
python report_generator.py --format markdown

# Session Vergleich
python report_generator.py --compare-sessions \
    benchmark_results_20240101 \
    benchmark_results_20240115
```

## 📋 Konfiguration

### model_config.yaml

Die Datei `model_config.yaml` enthält:
- Modell-Definitionen mit erwarteten Memory-Anforderungen
- Quantisierungs-Konfigurationen
- Benchmark-Presets (quick, standard, comprehensive)
- GPU-spezifische Einstellungen
- HuggingFace Hub Settings

```yaml
# Beispiel: Custom Benchmark
benchmark:
  my_test:
    models:
      - "mistralai/Mistral-7B-v0.1"
      - "meta-llama/Llama-2-7b-hf"
    quantizations: ["fp16", "int8"]
    batch_sizes: [1, 4, 8]
    iterations: 10
    warmup: 3
```

## 📊 Output-Struktur

```
benchmark_results/
├── gpt2_fp16_20240107_143022.json
├── mistral-7b_int8_20240107_144533.json
└── ...

reports/
├── report_20240107_150000.json
├── report_20240107_150000.csv
├── report_20240107_150000.md
├── viz_20240107_150000_throughput.png
├── viz_20240107_150000_memory.png
└── viz_20240107_150000_power_efficiency.png
```

## 🔧 Erweiterte Features

### 1. Memory Fragmentation Tracking

Das System trackt automatisch GPU Memory Fragmentation:
```python
fragmentation = (reserved_memory - allocated_memory) / reserved_memory
```

### 2. Power Efficiency Analysis

Berechnet Tokens pro Watt für jede Konfiguration:
```python
efficiency = tokens_per_second / power_consumption_watts
```

### 3. Batch Size Scaling

Misst wie gut ein Modell mit größeren Batches skaliert:
```python
scaling_efficiency = actual_throughput / expected_linear_throughput
```

### 4. Cache Hit Rates

Analysiert KV-Cache Effizienz (wenn verfügbar).

## 📈 Benchmark-Ergebnisse Interpretieren

### Markdown Report Sections

1. **Executive Summary**: Überblick über getestete Modelle
2. **Performance Highlights**: Best-of-Kategorie Winners
3. **Detailed Results**: Vollständige Tabellen pro Modell
4. **Batch Scaling Analysis**: Skalierungs-Effizienz
5. **Memory Analysis**: Speicherverbrauch Details
6. **Power & Thermal**: Energie-Effizienz
7. **Recommendations**: Automatische Empfehlungen

### Wichtige Metriken

| Metrik | Beschreibung | Ziel |
|--------|--------------|------|
| **Throughput** | Tokens/Sekunde | Höher ist besser |
| **Latency** | Zeit pro Iteration (ms) | Niedriger ist besser |
| **Memory Peak** | Maximaler VRAM-Verbrauch | Unter 45GB für L40S |
| **Fragmentation** | Verschwendeter Speicher | < 10% optimal |
| **Power Efficiency** | Tokens/s/Watt | Höher ist besser |
| **Scaling Efficiency** | Batch-Skalierung | > 90% gut |

## 🎓 Best Practices

### Für L40S (48GB VRAM)

1. **7B Modelle**:
   - FP16: Batch 8-16 optimal
   - INT8: Batch 16-32 möglich
   - INT4: Batch 32+ möglich

2. **13B Modelle**:
   - FP16: Nur kleine Batches (1-4)
   - INT8: Batch 8-16 empfohlen ✅
   - INT4: Batch 16-32 optimal

3. **34B Modelle**:
   - FP16: Passt nicht
   - INT8: Möglich mit Batch 1-4
   - INT4: Batch 4-8 empfohlen ✅

### Quantisierungs-Empfehlungen

- **Entwicklung/Debugging**: FP16 (beste Accuracy)
- **Production Serving**: INT8 (guter Kompromiss)
- **Maximum Throughput**: INT4 (akzeptable Quality)
- **Research**: BF16 (numerische Stabilität)

## 🔬 Beispiel-Workflows

### Workflow 1: Model Selection

```bash
# 1. Quick Test mehrerer Modelle
python llm_benchmark.py --quick

# 2. Detailed Test der Top 3
for model in mistral llama2-7b codellama-7b; do
    python llm_benchmark.py --model $model --quantization fp16 int8
done

# 3. Report generieren und vergleichen
python report_generator.py --plots
```

### Workflow 2: Optimization

```bash
# 1. Baseline mit FP16
python llm_benchmark.py --model mistralai/Mistral-7B-v0.1 \
    --quantization fp16 --batch-sizes 1 2 4 8 16

# 2. INT8 Quantisierung
python llm_benchmark.py --model mistralai/Mistral-7B-v0.1 \
    --quantization int8 --batch-sizes 1 2 4 8 16 32

# 3. Vergleich
python report_generator.py --format markdown
```

### Workflow 3: Production Planning

```bash
# 1. Test mit realistischen Batch Sizes
python llm_benchmark.py --model meta-llama/Llama-2-13b-hf \
    --quantization int8 --batch-sizes 4 8 16

# 2. Monitor während Benchmark
# Terminal 1:
python llm_benchmark.py ...

# Terminal 2:
python gpu_monitor.py

# 3. Analyse
python report_generator.py --plots
```

## 🐛 Troubleshooting

### Out of Memory Errors

```bash
# Lösung 1: Kleinere Batch Size
python llm_benchmark.py --model ... --batch-sizes 1 2

# Lösung 2: Stärkere Quantisierung
python llm_benchmark.py --model ... --quantization int4

# Lösung 3: GPU Cache leeren
python -c "import torch; torch.cuda.empty_cache()"
```

### Model Download Fehler

```bash
# HuggingFace Token setzen
export HF_TOKEN="your_token_here"

# Oder Login
huggingface-cli login

# Cache Verzeichnis setzen
export TRANSFORMERS_CACHE="./models_cache"
```

### CUDA Errors

```bash
# CUDA Version prüfen
nvidia-smi

# PyTorch CUDA Support prüfen
python -c "import torch; print(torch.cuda.is_available())"

# Neuinstallation mit korrekter CUDA Version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Weitere Ressourcen

- [HuggingFace Model Hub](https://huggingface.co/models)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [NVIDIA L40S Specs](https://www.nvidia.com/en-us/data-center/l40s/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 🤝 Contributing

Contributions sind willkommen! Bitte:
1. Fork das Repository
2. Erstelle einen Feature Branch
3. Teste deine Änderungen
4. Submit einen Pull Request

## 📄 Lizenz

[Ihre Lizenz hier]

## 🙏 Credits

Entwickelt für NVIDIA L40S GPU Performance Testing

---

**Letzte Aktualisierung**: 2025-01-07
**Version**: 1.0.0
