# 🎓 LLM Benchmark - Was passiert während der Ausführung?

Diese Datei erklärt **Schritt für Schritt**, was während `python llm_benchmark.py` passiert und was die Metriken bedeuten.

---

## 📋 Übersicht: Die 6 Tests

```
TEST 1/6: GPT-2 (124M) - FP16     → Baseline (volle Präzision)
TEST 2/6: GPT-2 (124M) - 8-bit    → 50% weniger Memory
TEST 3/6: GPT-2 (124M) - 4-bit    → 75% weniger Memory
TEST 4/6: GPT-2-Medium (355M) - FP16
TEST 5/6: GPT-2-Medium (355M) - 8-bit
TEST 6/6: GPT-2-Medium (355M) - 4-bit
```

**Jeder Test durchläuft 5 Phasen:**

---

## 🔄 Die 5 Phasen eines Tests

### **Phase 1: Model Loading** 📦

```
============================================================
  gpt2 - FP16
============================================================

📦 Loading model...
```

**Was passiert:**
1. **Tokenizer laden**: Lädt Vocabulary und Encoding-Regeln
2. **Model Download**: Beim ersten Mal von HuggingFace-Servern
3. **GPU Transfer**: Model-Weights werden in VRAM kopiert
4. **Initialisierung**: CUDA Kernels werden vorbereitet

**Code (vereinfacht):**
```python
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Model mit FP16
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,  # 16-bit Floating Point
    device_map="auto"            # Automatisch auf GPU
)
```

**Output:**
```
✅ Loaded in 2.15s
💾 GPU Memory: 0.500 GB
```

**Gemessene Metriken:**
- **Load Time**: Zeit von Start bis Model bereit (Sekunden)
- **GPU Memory**: VRAM-Verbrauch (GB)

**Typische Werte (L40S):**
```
GPT-2 (124M):
  FP16:  2-3s, 0.5 GB
  8-bit: 2-4s, 0.25 GB
  4-bit: 3-5s, 0.13 GB

GPT-2-Medium (355M):
  FP16:  3-5s, 1.4 GB
  8-bit: 4-6s, 0.7 GB
  4-bit: 5-7s, 0.35 GB
```

**Warum dauert Quantisierung länger beim Laden?**
- FP16: Weights werden direkt geladen (einfach)
- 8-bit/4-bit: Weights müssen quantisiert werden (extra Arbeit)

**Memory-Berechnung:**
```
GPT-2 hat 124 Millionen Parameter

FP16:  124M × 2 Bytes = 248 MB  (~0.25 GB roh)
       + Activations, Buffers    ≈ 0.5 GB total

8-bit: 124M × 1 Byte  = 124 MB  (~0.13 GB roh)
       + Overhead                 ≈ 0.25 GB total

4-bit: 124M × 0.5 Bytes = 62 MB (~0.06 GB roh)
       + Overhead                 ≈ 0.13 GB total
```

---

### **Phase 2: First Token Latency Test** 🔥

```
🔥 Testing first token latency...
⚡ First Token Latency: 8.45ms
```

**Was passiert:**
1. Prompt wird tokenisiert: `"The future of AI is"` → `[464, 2003, 286, 9552, 318]`
2. Input wird auf GPU kopiert
3. Model generiert **nur 1 Token**
4. Zeit wird gemessen: Start → erstes Token fertig

**Code:**
```python
# Tokenisiere Prompt
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

# Starte Timer
torch.cuda.synchronize()  # Wichtig: GPU-Operationen abwarten
start = time.time()

# Generiere NUR 1 Token
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False  # Greedy = deterministisch
    )

# Stoppe Timer
torch.cuda.synchronize()  # Wieder abwarten!
latency_ms = (time.time() - start) * 1000
```

**Warum ist `torch.cuda.synchronize()` wichtig?**
- GPU-Operationen laufen **asynchron**
- Python geht weiter, während GPU noch arbeitet
- `synchronize()` wartet, bis GPU wirklich fertig ist
- Ohne: Messung wäre falsch (zu schnell)!

**Output:**
```
⚡ First Token Latency: 8.45ms
```

**Interpretation:**
```
<10ms:     Sehr gut (gefühlt instant)
10-20ms:   Gut (kaum merkbar)
20-50ms:   OK (minimal spürbar)
50-100ms:  Mäßig (spürbar)
>100ms:    Langsam (störend)
```

**Warum ist das wichtig?**

**Szenario: Chat-Anwendung**
```
User: "Erkläre mir Quantencomputing"
       ↓
[First Token Latency = User wartet]
       ↓
System: "Quantencomputing..." ← Erste Zeichen erscheinen
```

**User Experience:**
- 10ms: "Wow, instant!"
- 50ms: "Schnell"
- 200ms: "Hmm, lädt..."
- 500ms: "Ist das kaputt?"

**Typische Werte (L40S):**
```
GPT-2:
  FP16:  8-12ms
  8-bit: 10-15ms
  4-bit: 12-18ms

GPT-2-Medium:
  FP16:  15-25ms
  8-bit: 18-30ms
  4-bit: 20-35ms
```

---

### **Phase 3: Warmup** 🏃‍♂️

```python
# Warmup (wichtig für faire Messungen!)
print("Warming up...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=10)
```

**Was passiert:**
- Model generiert 10 "Wegwerf-Tokens"
- GPU wird "warm"
- Caches werden gefüllt
- CUDA Kernels werden kompiliert (JIT)

**Warum notwendig?**

**Kalte GPU (ohne Warmup):**
```
Token 1:  50ms  ← Langsam! (Kernel kompilieren)
Token 2:  8ms   ← Schneller (Cache warm)
Token 3:  8ms
Token 4:  8ms
...
```

**Mit Warmup:**
```
Warmup: 10 Tokens (weggeworfen)
       ↓ GPU ist jetzt "warm"
Messung:
Token 1:  8ms   ← Konsistent!
Token 2:  8ms
Token 3:  8ms
...
```

**Wie ein Auto:**
- Motor startet kalt → schlechte Performance
- Motor warmlaufen lassen
- Dann erst Gas geben für faire Messung

**CUDA JIT Compilation:**
```
Erstes mal model.generate():
  → CUDA Kernel wird kompiliert (langsam)
  → Kernel wird gecached

Zweites mal:
  → Cached Kernel wird verwendet (schnell!)
```

**Diese Warmup-Phase wird NICHT gemessen!**

---

### **Phase 4: Throughput Test** 🚀

```
🚀 Testing throughput (50 tokens)...
✅ Throughput: 105.3 tokens/sec
⏱️  Total time: 0.47s
```

**Was passiert:**
1. Model generiert **50 neue Tokens**
2. Gesamtzeit wird gemessen
3. Throughput = Tokens ÷ Zeit

**Code:**
```python
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,      # 50 Tokens generieren
        do_sample=True,          # Sampling (realistisch)
        temperature=0.8,         # Kreativität
        top_p=0.9                # Nucleus Sampling
    )

torch.cuda.synchronize()
total_time = time.time() - start
throughput = 50 / total_time
```

**Output:**
```
✅ Throughput: 105.3 tokens/sec
⏱️  Total time: 0.47s

📝 Generated text preview:
   The future of artificial intelligence is very bright,
   and it will be exciting to see how these new technologies
   can be applied to solve problems that have been solved...
```

**Berechnung:**
```
50 Tokens in 0.47 Sekunden
→ 50 ÷ 0.47 = 106.4 tokens/sec
```

**Warum ist Throughput wichtig?**

**Szenario 1: Einzelner User (Chat)**
```
User fragt nach 200 Tokens Antwort

FP16 (100 tok/s):  200 ÷ 100 = 2.0 Sekunden
4-bit (150 tok/s): 200 ÷ 150 = 1.3 Sekunden

→ 0.7 Sekunden gespart (bessere UX!)
```

**Szenario 2: Batch Processing**
```
1000 Dokumente zusammenfassen, je 500 Tokens

FP16:  (1000 × 500) ÷ 100 = 5000 Sekunden = 83 Minuten
4-bit: (1000 × 500) ÷ 150 = 3333 Sekunden = 56 Minuten

→ 27 Minuten gespart!
```

**Typische Werte (L40S):**
```
GPT-2 (124M):
  FP16:  80-120 tok/s  (Baseline)
  8-bit: 100-150 tok/s (+25%)
  4-bit: 120-180 tok/s (+50%)

GPT-2-Medium (355M):
  FP16:  50-80 tok/s
  8-bit: 70-100 tok/s  (+40%)
  4-bit: 90-120 tok/s  (+60%)

Mistral 7B:
  FP16:  30-50 tok/s
  8-bit: 40-70 tok/s
  4-bit: 60-90 tok/s   (+80%)
```

---

### **Phase 5: Memory Cleanup** 🧹

```python
# Cleanup
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
```

**Was passiert:**
1. **del model**: Python-Referenz wird gelöscht
2. **gc.collect()**: Garbage Collector räumt Python-Objekte auf
3. **torch.cuda.empty_cache()**: GPU-Cache wird geleert

**Warum notwendig?**

**Ohne Cleanup:**
```
Test 1: Lädt GPT-2 FP16      → 0.5 GB belegt
Test 2: Lädt GPT-2 8-bit     → 0.75 GB belegt (0.5 + 0.25)
Test 3: Lädt GPT-2 4-bit     → 0.88 GB belegt (0.5 + 0.25 + 0.13)
...
Test 6: OUT OF MEMORY! ❌
```

**Mit Cleanup:**
```
Test 1: GPT-2 FP16  → 0.5 GB  → Cleanup → 0 GB
Test 2: GPT-2 8-bit → 0.25 GB → Cleanup → 0 GB
Test 3: GPT-2 4-bit → 0.13 GB → Cleanup → 0 GB
...
Test 6: Funktioniert! ✅
```

**Verifikation:**
```python
print(f"Memory before: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
# ... test ...
del model
gc.collect()
torch.cuda.empty_cache()
print(f"Memory after: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# Output:
# Memory before: 1.42 GB
# Memory after: 0.01 GB  ✅
```

---

## 📊 Am Ende: Die Zusammenfassung

Nach allen 6 Tests siehst du:

```
============================================================
  BENCHMARK SUMMARY
============================================================

Model                Quant    Load(s)    Mem(GB)    FTL(ms)    Throughput    Status
----------------------------------------------------------------------------------
gpt2                 FP16     2.15       0.500      8.45       105.3         ✅
gpt2                 8-bit    2.43       0.253      10.12      127.8         ✅
gpt2                 4-bit    2.87       0.130      12.34      155.2         ✅
gpt2-medium          FP16     3.21       1.420      15.67      72.4          ✅
gpt2-medium          8-bit    3.65       0.715      18.23      94.1          ✅
gpt2-medium          4-bit    4.12       0.362      21.45      118.7         ✅
```

### **Vergleichsanalyse:**

```
============================================================
  QUANTIZATION COMPARISON
============================================================

📊 gpt2 Comparison (vs FP16 baseline):

Quantization    Memory Saved    Speedup      FTL Change
-------------------------------------------------------
FP16            0.0%           1.00x        +0.0%
8-bit           49.4%          1.21x        +19.8%
4-bit           74.0%          1.47x        +46.0%
```

**Was bedeutet das?**

**Memory Saved:**
- 8-bit spart 49.4% VRAM → Kannst 2x mehr Modelle laden
- 4-bit spart 74.0% VRAM → Kannst 4x mehr Modelle laden

**Speedup:**
- 8-bit ist 1.21x schneller (21% Boost)
- 4-bit ist 1.47x schneller (47% Boost!)

**FTL Change:**
- 8-bit: 19.8% langsamer beim ersten Token
- 4-bit: 46% langsamer
- Aber immer noch <20ms = instant für User!

---

## 🔬 Warum ist 4-bit schneller?

### **Der GPU Memory Bandwidth Bottleneck**

**NVIDIA L40S Specs:**
```
Memory Bandwidth: 864 GB/s
Compute Power:    91.6 TFLOPS (FP32)
```

**Problem:** Daten-Transfer ist langsamer als Berechnung!

**Beispiel:**
```
Matrix Multiplication: A × B = C

Compute Time:  0.1 ms  (sehr schnell!)
Memory Load:   1.0 ms  (Flaschenhals!)
              ↑
         Hier warten wir 90% der Zeit!
```

**Mit Quantisierung:**

**FP16 (jedes Weight = 2 Bytes):**
```
124M Parameter × 2 Bytes = 248 MB

Transfer Zeit: 248 MB ÷ 864 GB/s = 0.287 ms
Compute Zeit:  0.1 ms
Total:         0.387 ms
```

**4-bit (jedes Weight = 0.5 Bytes):**
```
124M Parameter × 0.5 Bytes = 62 MB

Transfer Zeit: 62 MB ÷ 864 GB/s = 0.072 ms  ← 4x schneller!
De-Quant Zeit: 0.05 ms
Compute Zeit:  0.1 ms
Total:         0.222 ms

→ 1.74x schneller als FP16!
```

**Visualisierung:**

```
FP16:  ████████ (Memory Transfer)
       ███ (Compute)
       Total: ███████████

4-bit: ██ (Memory Transfer)
       █ (De-Quantization)
       ███ (Compute)
       Total: ██████

→ 4-bit ist kürzer = schneller!
```

**Faustregel:**
- Kleine Modelle: Memory Bandwidth limitiert
- Große Modelle: Compute limitiert
- L40S mit LLMs: Fast immer Memory-bound
- → Quantisierung hilft massiv!

---

## 💾 Die JSON-Ausgabe

Am Ende wird `llm_benchmark_results.json` erstellt:

```json
{
  "results": [
    {
      "model_name": "gpt2",
      "quantization": "FP16",
      "load_time_sec": 2.15,
      "memory_gb": 0.500,
      "first_token_latency_ms": 8.45,
      "throughput_tokens_per_sec": 105.3,
      "total_inference_time_sec": 0.47,
      "num_tokens_generated": 50,
      "success": true
    },
    ...
  ],
  "metadata": {
    "gpu": "NVIDIA L40S",
    "cuda_version": "12.1",
    "pytorch_version": "2.0.1",
    "total_vram_gb": 47.7
  }
}
```

**Verwendung:**
- Visualisierung mit `visualize_results.py`
- Eigene Analyse mit Python/pandas
- Vergleich mit späteren Runs
- Dokumentation für Papers/Reports

---

## 🎯 Key Learnings

### **1. Quantisierung ist fast immer ein Gewinn**
```
Trade-offs:
  Memory: -50% (8-bit) bis -75% (4-bit)
  Speed:  +20% (8-bit) bis +50% (4-bit)
  Quality: -0.5% (8-bit) bis -2% (4-bit)

→ Lohnt sich fast immer!
```

### **2. First Token Latency ≠ Throughput**
```
Use Case entscheidet:
  Chat/Streaming:    FP16 (niedrige FTL)
  Batch Processing:  4-bit (hoher Throughput)
  Multi-Model:       4-bit (wenig Memory)
```

### **3. Warmup ist essentiell**
```
Ohne Warmup:
  Test 1: 50ms
  Test 2: 8ms
  Test 3: 8ms
  → Inkonsistent!

Mit Warmup:
  Test 1: 8ms
  Test 2: 8ms
  Test 3: 8ms
  → Fair & reproduzierbar!
```

### **4. Memory Bandwidth ist oft der Bottleneck**
```
Große Modelle = viel Daten transferieren
L40S Bandwidth: 864 GB/s

4-bit: 4x weniger Daten = 4x weniger Wartezeit!
```

### **5. Messen statt raten!**
```
"I think 8-bit is slower"
  ↓
Run benchmark
  ↓
"Wow, 8-bit is 21% faster!"

→ Daten schlagen Intuition!
```

---

## 🚀 Nächste Schritte

Nach dem Benchmark:

1. **Analysiere die Ergebnisse**
   ```bash
   cat llm_benchmark_results.json | python -m json.tool
   ```

2. **Erstelle Visualisierungen**
   ```bash
   python visualize_results.py
   ```

3. **Experimentiere**
   - Teste größere Modelle (Mistral 7B)
   - Ändere Prompt-Länge
   - Teste verschiedene Batch-Sizes

4. **Optimiere für deinen Use Case**
   - Chat → FP16 oder 8-bit
   - Batch → 4-bit
   - Multi-Model → 4-bit

---

**Happy Benchmarking! 🎉**
