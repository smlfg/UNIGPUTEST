# 🎬 VIDEO GENERATION PROJECT – Claude Code GPU Task

**Dein Setup:** NVIDIA L40S (48 GB VRAM), fact-gpt VM, UNIGPUTEST Repository

**Ziel:** Production-Ready Video-Generierung mit Stable Diffusion / Stable Video Diffusion auf der L40S GPU

---

## 📊 SITUATION

Du arbeitest auf der `fact-gpt.stud.it.hs-worms.de` VM mit:
- **GPU:** NVIDIA L40S (48 GB VRAM, 91.6 TFLOPS, CUDA 13.0)
- **CPU:** 32 Kerne, 64 GB RAM
- **Storage:** 1 TB
- **Assets:** 33 Video-Segmente in `KIVIDEO/data/veo_segments.json`
- **Code:** Production-Ready `video_generator.py` (Ollama + ComfyUI Integration)

---

## 🎯 HAUPTAUFGABE

### Option A: GPU-Accelerated Stable Diffusion Video Pipeline (3-4h)

Erstelle ein **Production-Ready System**, das:

#### 1. **Stable Diffusion Text-to-Image Generation**
- Nutze `diffusers` Library (Hugging Face)
- Model: `stabilityai/stable-diffusion-2-1` oder `stabilityai/sdxl-turbo`
- GPU-Accelerated Inference mit PyTorch CUDA
- Batch Processing für mehrere Segmente parallel

#### 2. **Frame-to-Video Compilation**
- Generiere 24 Frames pro Segment (8 Sekunden @ 3 FPS)
- FFmpeg H.264/HEVC Encoding (GPU-accelerated wenn möglich)
- 1080p Output mit konsistenten Lichtverhältnissen

#### 3. **Segment Processing Pipeline**
```python
# Workflow:
veo_segments.json → Segment Prompts
    ↓
[Text-to-Image] Stable Diffusion (GPU)
    ↓
[Frame Generation] 24 Frames pro Segment
    ↓
[Video Compilation] FFmpeg MP4 Encoding
    ↓
[Quality Check] VRAM Usage, Latency, Konsistenz
    ↓
segment_01.mp4, segment_02.mp4, ..., segment_33.mp4
```

#### 4. **GPU Memory Management**
- Monitor VRAM Usage mit `nvidia-smi`
- Automatic Model Offloading bei >40 GB Usage
- Thermal Monitoring (Pause bei >85°C)
- Batch Size Optimization

#### 5. **Character & Object Consistency** (Kritisch!)
- **Protagonist:** Mann mit blauem Hemd (Segmente 1, 4, 5, 8, 9, 11, 13, 22, 26-29)
- **Zweiter Mann:** Hut + Bart (Segmente 16, 22-28, 30, 33)
- **Apfelbaum:** Setzling 1.2-1.5m (Segmente 1-22), Gewachsener Baum 3-4m (Segmente 24-33)
- **Location:** Feld/Acker mit hellem Tageslicht (5500-6000K)

**Consistency Strategy:**
- Nutze **gleiche Seeds** für gleiche Charaktere (z.B. Protagonist = Seed 1000-Serie)
- Nutze **ControlNet** oder **IP-Adapter** für Character Reference
- Alternativ: **SDXL Turbo** mit Prompt-Embeddings für Konsistenz

---

## 📋 CODE-ANFORDERUNGEN

### Must-Have Features:
```python
1. **Modular Pipeline Architecture**
   - `generate_single_segment(prompt, seed, output_path)`
   - `batch_generate_segments(segments_json, output_dir)`
   - `compile_final_video(segment_videos, output_path)`

2. **GPU Monitoring & Safety**
   - Real-time VRAM Tracking
   - Thermal Throttling (Pause bei >85°C)
   - Automatic Memory Cleanup zwischen Segmenten

3. **Progress Tracking**
   - Terminal Progress Bar (tqdm)
   - ETA Calculation
   - Per-Segment Timing Report

4. **Error Recovery**
   - Checkpoint/Resume Capability
   - Automatic Re-Generation bei Fehlern
   - Graceful Degradation (Fallback zu niedrigerer Auflösung)

5. **Quality Assurance**
   - Continuity Score (Character Consistency Check)
   - Frame Quality Check (Resolution, Aspect Ratio)
   - Segment Transition Analysis
```

### Best Practices:
- **PyTorch Mixed Precision (FP16)** für 2x Speed-up
- **Sequential Processing** (nicht parallel) für Konsistenz
- **Deterministische Seeds** für Reproduzierbarkeit
- **Detailed Logging** für Debugging

---

## 🚀 DELIVERABLES

### Code:
1. **Main Pipeline Script** (`sd_video_pipeline.py`)
   - CLI Interface mit argparse
   - Single Segment Mode
   - Batch Mode (alle 33 Segmente)
   - Config System (YAML/JSON)

2. **GPU Benchmark Script** (`benchmark_sd_video.py`)
   - Latency pro Segment
   - VRAM Usage Peak/Sustained
   - Thermal Performance
   - Quality Scores

3. **Test Cases** (`test_video_pipeline.py`)
   - Unit Tests für jede Pipeline-Phase
   - Integration Tests
   - Performance Tests

### Documentation:
1. **Setup Guide** (Installation, Dependencies)
2. **Usage Examples** (Single Segment, Batch, Custom Prompts)
3. **Troubleshooting Guide** (Common Errors, Solutions)
4. **Performance Report** (Benchmarks, Metrics, Charts)

### Outputs:
1. **Generated Segments** (33x MP4 @ 8 Sekunden, 1080p)
2. **Final Compilation** (`apfelbaum_video.mp4`, 4:24 Minuten)
3. **Generation Report** (JSON: timing, VRAM, quality scores)
4. **Continuity Analysis** (Character consistency scores)

---

## 🎬 KRITISCHE HERAUSFORDERUNGEN

### 1. **Character Consistency Problem**
**Problem:** Stable Diffusion generiert jedes Frame independent → Charaktere sehen unterschiedlich aus

**Solutions:**
- **Option 1:** ControlNet + Reference Image
  ```python
  # Generiere einmal "Mann mit blauem Hemd"
  reference_image = generate_character_reference(
      "man in blue shirt, documentary style",
      seed=1000
  )

  # Nutze dieses Image als ControlNet Input für alle Segmente
  for segment in protagonist_segments:
      frames = generate_with_controlnet(
          prompt=segment.prompt,
          control_image=reference_image,
          seed=1000 + segment.id
      )
  ```

- **Option 2:** SDXL Turbo + Embedding Consistency
  ```python
  # Nutze gemeinsame Token-Embeddings
  protagonist_embedding = encode_prompt(
      "young man in blue shirt, short hair, casual"
  )

  # Inject Embedding in alle Protagonist-Prompts
  ```

- **Option 3:** Frame Interpolation (Post-Processing)
  ```python
  # Nutze Optical Flow für smooth Transitions
  from film_net import interpolate_frames
  smooth_frames = interpolate_frames(generated_frames)
  ```

### 2. **Location Consistency**
**Problem:** Feld/Acker muss durchgängig gleich aussehen

**Solution:**
- Nutze **Location-Seed** (z.B. 5000) für alle Feld-Segmente
- Prefix in Prompt: `"In the same bright sunny farmland field, ..."`
- ControlNet mit statischem Background-Image

### 3. **Lighting Consistency**
**Problem:** Tageslicht muss konsistent sein (5500-6000K)

**Solution:**
- Prompt-Prefix: `"Bright natural daylight, 5500K color temperature, outdoor lighting, ..."`
- Post-Processing: Color Grading mit FFmpeg
  ```bash
  ffmpeg -i input.mp4 -vf "colorbalance=rs=0.1:gs=0.1:bs=-0.1" output.mp4
  ```

### 4. **Transformation Segment 23→24**
**Problem:** Setzling (1.2m) → Gewachsener Baum (3-4m)

**Solution:**
- Generiere 2 separate Frames
- Nutze **Morphing/Dissolve Transition** in FFmpeg
  ```bash
  ffmpeg -i segment_23.mp4 -i segment_24.mp4 \
         -filter_complex "[0][1]blend=all_expr='A*(1-T/1)+B*(T/1)'" \
         transition.mp4
  ```

---

## 🔧 TECHNISCHE SPEZIFIKATIONEN

### Stable Diffusion Setup:
```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Model Loading
model_id = "stabilityai/stable-diffusion-2-1"  # oder sdxl-turbo
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # FP16 für Speed
    safety_checker=None,        # Disable für Performance
)
pipe = pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

# Enable Memory Optimization
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Generate Frame
image = pipe(
    prompt="...",
    num_inference_steps=30,      # 30 für Quality, 20 für Speed
    guidance_scale=7.5,          # Creativity vs Accuracy
    height=1080,
    width=1920,
    generator=torch.manual_seed(seed),
).images[0]
```

### GPU Monitoring:
```python
import subprocess
import re

def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    memory_mb = int(result.stdout.strip())
    return memory_mb / 1024  # Convert to GB

def get_gpu_temperature():
    """Get current GPU temperature in Celsius."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

# Use in Pipeline:
def safe_generate(prompt, seed):
    temp = get_gpu_temperature()
    if temp > 85:
        print(f"⚠️  GPU Temperature {temp}°C > 85°C. Pausing...")
        time.sleep(60)

    vram_before = get_gpu_memory()
    image = pipe(prompt, generator=torch.manual_seed(seed)).images[0]
    vram_peak = get_gpu_memory()

    print(f"VRAM: {vram_before:.1f}GB → {vram_peak:.1f}GB")
    return image
```

### FFmpeg Video Compilation:
```python
def compile_frames_to_video(frame_dir, output_path, fps=3):
    """Compile PNG frames to MP4 video."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", f"{frame_dir}/*.png",
        "-c:v", "libx264",          # H.264 Codec
        "-pix_fmt", "yuv420p",      # Compatibility
        "-crf", "18",               # Quality (18 = Very High)
        "-preset", "slow",          # Encoding Speed vs Quality
        "-vf", "scale=1920:1080",   # Force 1080p
        output_path
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✅ Video compiled: {output_path}")
```

---

## 🎯 PERFORMANCE TARGETS

### Per-Segment Generation:
- **Latency:** <60 seconds pro Segment (24 Frames @ 30 steps)
- **VRAM Usage:** <40 GB peak (leave 8 GB buffer)
- **Quality:** 1080p, sharp focus, consistent colors
- **Continuity Score:** >0.85 (Character/Object Consistency)

### Batch Processing:
- **Total Time:** <2 Stunden für 33 Segmente
- **Success Rate:** >95% (max 2 Re-Generations)
- **Final Video:** 4:24 Minuten, 1080p, H.264, <500 MB

### Thermal Management:
- **Max Temperature:** 85°C (Pause wenn überschritten)
- **Resume Temperature:** 80°C
- **Monitoring Interval:** 10 Sekunden

---

## 🚀 NEXT STEPS

### Phase 1: Setup & Testing (30 min)
```bash
# Install Dependencies
conda activate user30_ml
pip install diffusers transformers accelerate safetensors pillow tqdm

# Test Stable Diffusion
python -c "
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-1',
    torch_dtype=torch.float16
).to('cuda')

image = pipe('man in blue shirt, documentary style', num_inference_steps=20).images[0]
image.save('test_output.png')
print('✅ Stable Diffusion working!')
"
```

### Phase 2: Single Segment Generation (1h)
- Implement `generate_single_segment()`
- Test mit Segment 01 (Intro)
- Validate Output Quality
- Benchmark VRAM + Latency

### Phase 3: Character Consistency (1h)
- Implement ControlNet or Embedding Strategy
- Generate Reference Images
- Test Consistency across 3 Segments
- Measure Continuity Score

### Phase 4: Batch Processing (1h)
- Implement `batch_generate_segments()`
- Process alle 33 Segmente sequentiell
- Monitor GPU Temperature
- Save Checkpoints

### Phase 5: Final Compilation (30 min)
- Concatenate alle Segment-Videos
- Add Transition Effects (Segment 23→24)
- Generate Final Report
- Quality Assurance

---

## 📊 SUCCESS CRITERIA

✅ **Funktionierender Code:**
- Pipeline läuft ohne Crashes
- Alle 33 Segmente generiert
- Final Video kompiliert

✅ **Performance:**
- <2 Stunden Total Time
- <40 GB VRAM Peak
- GPU Temperature <85°C

✅ **Quality:**
- 1080p Output
- Character Consistency >85%
- Smooth Transitions

✅ **Documentation:**
- Setup Guide vorhanden
- Benchmarks dokumentiert
- Code gut kommentiert

---

## 🎬 EXAMPLE USAGE

### Single Segment:
```bash
python sd_video_pipeline.py \
    --prompt "Medium shot of a man in blue shirt holding a document" \
    --seed 1001 \
    --output segment_01.mp4 \
    --frames 24 \
    --fps 3
```

### Batch Mode (Alle Segmente):
```bash
python sd_video_pipeline.py \
    --batch KIVIDEO/data/veo_segments.json \
    --master-prompt KIVIDEO/docs/Veo31MasterPrompt.md \
    --output-dir ./generated_videos \
    --concat-output apfelbaum_final.mp4
```

### Benchmark:
```bash
python benchmark_sd_video.py \
    --segments KIVIDEO/data/veo_segments.json \
    --output benchmark_report.json
```

---

## 💡 BONUS CHALLENGES (Optional)

### 1. **Audio Integration**
- Generate Narration mit TTS (Coqui TTS)
- Add Background Music
- Sync Audio mit Video-Segments

### 2. **Upscaling**
- Real-ESRGAN für 4K Upscaling
- Frame Interpolation für 30 FPS

### 3. **Snapdragon Export**
- Convert zu ONNX für Snapdragon X Elite
- Optimize für NPU Inference
- Mobile App mit Video Playback

---

## 🔥 LET'S START!

**Meine Empfehlung:** Starte mit **Phase 1 + 2** (Setup & Single Segment).

Wenn das funktioniert, erweitern wir zu Batch Processing + Consistency!

**Geschätzte Zeit:** 3-4 Stunden für Full Pipeline
**GPU-Auslastung:** 80-90% während Generation
**VRAM Usage:** 25-35 GB (Stable Diffusion 2.1)

🚀 **Ready to generate 33 AI video segments on the L40S GPU?**

---

## 📞 SUPPORT

Falls du stecken bleibst:
1. Zeige mir `nvidia-smi` Output
2. Zeige mir Error Messages
3. Zeige mir VRAM Usage während Generation

Ich helfe dir debuggen! 💪
