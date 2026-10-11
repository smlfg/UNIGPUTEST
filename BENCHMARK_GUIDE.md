# 📊 LLM Benchmark Suite - Vollständige Anleitung

Professionelles Benchmark-System für LLM Performance-Evaluierung auf NVIDIA L40S.

> 💡 **Neu hier?** Lies zuerst [BENCHMARK_EXPLAINED.md](BENCHMARK_EXPLAINED.md) für eine Schritt-für-Schritt Erklärung, was während des Benchmarks passiert!

---

## 🎯 Was wird gemessen?

### 1. **Load Time** (Sekunden)
**Was:** Zeit zum Laden des Modells in den GPU-Speicher
**Warum wichtig:**
- Cold-Start-Performance
- Serverless-Anwendungen
- Model-Switching bei Multi-Model-Serving

**Typische Werte:**
- GPT-2 (FP16): 1-3s
- GPT-2 (4-bit): 2-4s (Quantization-Overhead)
- Mistral 7B (4-bit): 20-40s

---

### 2. **GPU Memory** (GB)
**Was:** VRAM-Verbrauch des Modells
**Warum wichtig:**
- Bestimmt maximale Batch-Size
- Multi-Model-Deployment-Planung
- Kostenoptimierung (GPU-Auswahl)

**Typische Werte:**
```
GPT-2 (124M Parameter):
  FP16:  ~0.5 GB
  8-bit: ~0.25 GB  (50% Ersparnis)
  4-bit: ~0.13 GB  (75% Ersparnis)

GPT-2-Medium (355M):
  FP16:  ~1.4 GB
  8-bit: ~0.7 GB
  4-bit: ~0.35 GB

Mistral 7B (7B Parameter):
  FP16:  ~14 GB
  8-bit: ~7 GB
  4-bit: ~3.5 GB
```

**Faustregel:**
- 1B Parameter ≈ 2GB (FP16)
- 1B Parameter ≈ 1GB (8-bit)
- 1B Parameter ≈ 0.5GB (4-bit)

---

### 3. **First Token Latency** (Millisekunden)
**Was:** Zeit bis zum ersten generierten Token
**Warum wichtig:**
- User Experience in Chat-Anwendungen
- "Time to First Byte" (TTFB)
- Gefühlte Responsiveness

**Typische Werte:**
- GPT-2: 5-15ms
- GPT-2-Medium: 10-30ms
- Mistral 7B: 50-150ms

**Interpretation:**
- <50ms: Sehr gut (gefühlt instant)
- 50-200ms: Gut (kaum merkbar)
- >200ms: Spürbare Verzögerung

---

### 4. **Throughput** (Tokens/Sekunde)
**Was:** Generierungsgeschwindigkeit nach dem ersten Token
**Warum wichtig:**
- Gesamt-Inference-Geschwindigkeit
- Batch-Processing-Performance
- Kosten pro Token

**Typische Werte:**
```
NVIDIA L40S (48GB):
  GPT-2:
    FP16:  80-120 tok/s
    8-bit: 100-150 tok/s  (+20-30% schneller)
    4-bit: 120-180 tok/s  (+50% schneller)

  GPT-2-Medium:
    FP16:  50-80 tok/s
    8-bit: 70-100 tok/s
    4-bit: 90-120 tok/s

  Mistral 7B:
    FP16:  30-50 tok/s
    8-bit: 40-70 tok/s
    4-bit: 60-90 tok/s
```

**Faktoren:**
- Größere Modelle → langsamer
- Quantisierung → schneller (weniger Daten)
- Batch Size → höherer Throughput (bei größeren Batches)

---

### 5. **Efficiency Score** (Tokens/sec per GB)
**Was:** Throughput geteilt durch Memory-Verbrauch
**Warum wichtig:**
- Gesamt-Effizienz-Metrik
- Multi-Model-Serving-Planung
- Cost-Performance-Ratio

**Beispiel:**
```
GPT-2 FP16:  100 tok/s / 0.5 GB = 200 efficiency
GPT-2 4-bit: 150 tok/s / 0.13 GB = 1150 efficiency

→ 4-bit ist 5.75x effizienter!
```

---

## 🔬 Quantisierung verstehen

### **FP16 (16-bit Float)**
**Wie funktioniert's:**
- Jedes Gewicht: 16 Bits (2 Bytes)
- Volle Präzision der Original-Weights

**Vor- und Nachteile:**
- ✅ Höchste Qualität
- ✅ Numerisch stabil
- ❌ 2x Speicher von FP32
- ❌ Langsamer als quantisierte Varianten

**Wann verwenden:** Wenn Qualität wichtiger als Speed/Memory

---

### **INT8 (8-bit Integer)**
**Wie funktioniert's:**
- Jedes Gewicht: 8 Bits (1 Byte)
- Dynamische Quantisierung: Weights werden zur Laufzeit konvertiert
- Verwendet LLM.int8() von bitsandbytes

**Mathematik:**
```python
# Vereinfacht:
weight_fp16 = 0.753  # Original
scale = max(abs(weights)) / 127  # Berechne Skalierung
weight_int8 = round(weight_fp16 / scale)  # -128 bis 127

# Beim Inference:
weight_fp16_approx = weight_int8 * scale
```

**Vor- und Nachteile:**
- ✅ 50% weniger Memory
- ✅ ~1.5-2x schneller
- ✅ Minimaler Qualitätsverlust (<1%)
- ❌ Leichter Overhead durch De-Quantisierung

**Wann verwenden:** Gute Balance - Standard-Choice für Production

---

### **INT4 (4-bit Integer)**
**Wie funktioniert's:**
- Jedes Gewicht: 4 Bits (0.5 Bytes!)
- Aggressive Quantisierung mit Gruppierung
- Verwendet GPTQ oder QLoRA

**Trick:**
- Weights werden in Gruppen quantisiert (z.B. 128 Weights zusammen)
- Jede Gruppe hat eigene Skalierung → weniger Fehler

**Vor- und Nachteile:**
- ✅ 75% weniger Memory
- ✅ ~2-4x schneller
- ✅ Nur 1-3% Qualitätsverlust
- ❌ Mehr Quantisierungs-Overhead
- ❌ Nicht alle Operationen 4-bit

**Wann verwenden:** Für große Modelle, oder wenn Memory knapp ist

---

## 🚀 Verwendung

### **Schritt 1: Benchmark ausführen**

```bash
# Standard-Benchmark (GPT-2 Modelle)
python llm_benchmark.py

# Dauert: ~5-10 Minuten
# Testet: GPT-2, GPT-2-Medium mit FP16, 8-bit, 4-bit
```

**Output:**
```
TEST 1/6
============================================================
  gpt2 - FP16
============================================================

📦 Loading model...
✅ Loaded in 2.15s
💾 GPU Memory: 0.500 GB

🔥 Testing first token latency...
⚡ First Token Latency: 8.45ms

🚀 Testing throughput (50 tokens)...
✅ Throughput: 105.3 tokens/sec
⏱️  Total time: 0.47s

📝 Generated text preview:
   The future of artificial intelligence is very bright...

✅ Benchmark completed successfully!
```

---

### **Schritt 2: Ergebnisse visualisieren**

```bash
python visualize_results.py
```

**Erstellt 4 Plots:**
1. `memory_comparison.png` - Memory-Verbrauch
2. `throughput_comparison.png` - Geschwindigkeit
3. `latency_comparison.png` - First Token Latency
4. `efficiency_heatmap.png` - Effizienz-Matrix

---

### **Schritt 3: Ergebnisse analysieren**

```bash
# JSON ansehen
cat llm_benchmark_results.json | python -m json.tool

# Oder mit jq (wenn installiert):
jq '.results[] | {model: .model_name, quant: .quantization, memory: .memory_gb, throughput: .throughput_tokens_per_sec}' llm_benchmark_results.json
```

---

## 🎓 Lern-Beispiele

### **Beispiel 1: Memory-Optimierung**

**Szenario:** Du willst mehrere Modelle gleichzeitig laden

```python
# L40S hat 48 GB VRAM

# Option 1: FP16
# GPT-2-Medium: 1.4 GB × 3 Modelle = 4.2 GB ✅
# Mistral 7B: 14 GB × 3 = 42 GB ✅ (knapp!)

# Option 2: 4-bit
# GPT-2-Medium: 0.35 GB × 10 Modelle = 3.5 GB ✅
# Mistral 7B: 3.5 GB × 12 Modelle = 42 GB ✅

# → 4-bit erlaubt 3-4x mehr Modelle!
```

---

### **Beispiel 2: Latency vs Throughput Trade-off**

```
Ergebnis aus Benchmark:

GPT-2 FP16:
  First Token: 8ms
  Throughput: 100 tok/s

GPT-2 4-bit:
  First Token: 12ms  (+50% langsamer)
  Throughput: 150 tok/s  (+50% schneller)
```

**Interpretation:**
- **Chat-App (wenige Tokens):** FP16 besser (schnellere Response)
- **Batch-Generation (viele Tokens):** 4-bit besser (höherer Gesamtdurchsatz)

---

### **Beispiel 3: Cost-Efficiency**

```
AWS p4d.24xlarge (8× A100 80GB): $32.77/Stunde

Szenario: 1 Million Tokens pro Stunde generieren

FP16 Mistral 7B:
  40 tok/s × 3600s = 144k tokens/Stunde
  → Brauche 7 GPUs
  → Cost: $28.67/Stunde

4-bit Mistral 7B:
  80 tok/s × 3600s = 288k tokens/Stunde
  → Brauche 4 GPUs
  → Cost: $16.38/Stunde

→ 43% Kosteneinsparung durch Quantisierung!
```

---

## 🔧 Erweiterte Nutzung

### **Eigene Modelle hinzufügen**

Editiere `llm_benchmark.py`:

```python
models_to_test = [
    ("gpt2", "GPT-2 (124M)"),
    ("gpt2-medium", "GPT-2-Medium (355M)"),

    # Füge hinzu:
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B"),
    ("meta-llama/Llama-2-7b-hf", "Llama 2 7B"),  # Braucht HF Token!
]
```

---

### **Mehr Tokens testen**

```python
# In llm_benchmark.py, Zeile ~315:
num_tokens = 50  # Standard

# Ändere zu:
num_tokens = 100  # Für längere Generation
num_tokens = 200  # Für sehr lange Generation
```

**Trade-off:** Länger dauert, aber realistischere Throughput-Messung

---

### **Batch-Size testen**

```python
# Füge in benchmark_model() hinzu:
batch_prompts = [test_prompt] * batch_size
inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

# Dann:
outputs = model.generate(**inputs, max_new_tokens=num_tokens)
```

**Wichtig:** Batch Size erhöht Throughput, aber auch Memory!

---

## 📈 Erwartete Ergebnisse (L40S)

### **GPT-2 (124M)**
```
Quantization  Memory    FTL      Throughput
FP16         0.5 GB    8ms      100 tok/s
8-bit        0.25 GB   10ms     130 tok/s
4-bit        0.13 GB   12ms     160 tok/s
```

### **GPT-2-Medium (355M)**
```
Quantization  Memory    FTL      Throughput
FP16         1.4 GB    15ms     70 tok/s
8-bit        0.7 GB    18ms     95 tok/s
4-bit        0.35 GB   22ms     120 tok/s
```

### **Mistral 7B**
```
Quantization  Memory    FTL      Throughput
FP16         14 GB     80ms     35 tok/s
8-bit        7 GB      95ms     55 tok/s
4-bit        3.5 GB    110ms    75 tok/s
```

---

## 🐛 Troubleshooting

### **Out of Memory (OOM)**

```python
# Lösung 1: Kleineres Modell
models_to_test = [("gpt2", "GPT-2")]  # Statt gpt2-large

# Lösung 2: Weniger Quantisierungen
quantizations = [("4-bit", ...)]  # Nur 4-bit

# Lösung 3: Memory cleanup
torch.cuda.empty_cache()
gc.collect()
```

---

### **Langsame First Run**

**Grund:** Model-Download von HuggingFace

```bash
# Pre-download Modelle:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2-medium')"
```

---

### **Visualisierung funktioniert nicht**

```bash
# Installiere matplotlib:
pip install matplotlib

# Oder ohne Display (Server):
export MPLBACKEND=Agg
python visualize_results.py
```

---

## 📚 Weiterführende Ressourcen

- [Quantization Paper (LLM.int8())](https://arxiv.org/abs/2208.07339)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [bitsandbytes Library](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main_classes/quantization)

---

## 🎯 Key Takeaways

1. **4-bit Quantization ist oft die beste Wahl**
   - 75% weniger Memory
   - 2-3x schneller
   - Minimaler Qualitätsverlust

2. **First Token Latency ≠ Throughput**
   - Verschiedene Use Cases brauchen verschiedene Optimierungen

3. **Memory ist meist der Bottleneck**
   - Quantization erlaubt größere Modelle oder höhere Batch Sizes

4. **Messen ist wichtig!**
   - "In theory, theory and practice are the same. In practice, they're not."

---

**Viel Erfolg beim Benchmarking! 🚀**
