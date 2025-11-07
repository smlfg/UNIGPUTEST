# 📊 Monitoring Guide

## Real-Time Training Monitoring Tools

This project includes several tools to monitor your training in real-time.

---

## 🖥️ GPU Monitor

**Monitor GPU usage, memory, temperature, and utilization.**

### Basic Usage:

```bash
python src/utils/gpu_monitor.py
```

### With Logging:

```bash
# Log GPU stats to CSV every 5 seconds:
python src/utils/gpu_monitor.py --interval 5 --log gpu_stats.csv
```

### One-Time Check:

```bash
# Just check current stats once:
python src/utils/gpu_monitor.py --once
```

### Output Example:

```
🖥️  GPU MONITORING - Live Stats
═══════════════════════════════════════════════════════════════
GPU: NVIDIA L40S

💾 VRAM Usage:
   [████████████████░░░░░░░░░░░░░░░░░░░░░] 40.5%
   Used: 6.84 GB / 44.39 GB
   Free: 37.55 GB

⚡ GPU Utilization:
   [█████████████████████████████████████░] 98%

🌡️  Temperature: 🟢 45°C

⚡ Power Usage: 187.3W
```

---

## 📈 Training Dashboard

**Monitor training progress with live updates.**

### Usage:

```bash
# In one terminal - run training:
python src/training/train.py > training.log 2>&1

# In another terminal - monitor:
python src/utils/training_dashboard.py --log training.log
```

### With Custom Interval:

```bash
python src/utils/training_dashboard.py --log training.log --interval 10
```

### Output Example:

```
🔥 TRAINING DASHBOARD - Live Progress
══════════════════════════════════════════════════════════

📊 Current Progress:
   Epoch:         1.25 / 3.00
   Loss:          0.8734
   Learning Rate: 0.000175

Progress: [████████████████░░░░░░░░░░░░░░░░░░░░] 41.7%

📈 Loss Trend (last 10 steps): 📉 Decreasing
   Start: 0.9821 → Current: 0.8734

⏱️  Estimated Time Remaining: 1:23:45
⏱️  Elapsed Time: 0:45:12

📊 Statistics:
   Total Steps: 523
   Best Loss:   0.8234
   Worst Loss:  2.4512
```

---

## 📊 TensorBoard

**Advanced visualization with graphs and charts.**

### Start TensorBoard:

```bash
tensorboard --logdir checkpoints/logs --port 6006
```

### Open in Browser:

```
http://localhost:6006
```

### Features:

- 📈 **Scalars**: Loss, learning rate over time
- 🖼️ **Images**: (if logging images)
- 📊 **Histograms**: Weight distributions
- 🔍 **Graphs**: Model architecture

---

## 🎯 Recommended Setup

### During Training - 3 Terminals:

**Terminal 1: Training**
```bash
python src/training/train.py | tee training.log
```
*(Shows output + saves to file)*

**Terminal 2: GPU Monitor**
```bash
python src/utils/gpu_monitor.py --interval 2
```
*(Live GPU stats)*

**Terminal 3: Training Dashboard**
```bash
python src/utils/training_dashboard.py --log training.log --interval 5
```
*(Live training progress)*

---

## 🔔 Notifications (Optional)

### Get Notified When Training Completes:

```bash
# Linux with notify-send:
python src/training/train.py && notify-send "Training Complete!"

# With sound:
python src/training/train.py && (echo -e '\a' && notify-send "Training Complete!")

# Send to phone (using ntfy.sh):
python src/training/train.py && curl -d "Training complete!" ntfy.sh/your-topic
```

---

## 📝 Logging Best Practices

### 1. Always Save Training Logs:

```bash
python src/training/train.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

### 2. Log GPU Stats:

```bash
python src/utils/gpu_monitor.py --log logs/gpu_$(date +%Y%m%d_%H%M%S).csv &
```

### 3. Enable TensorBoard:

Already enabled by default! Logs saved to `checkpoints/logs/`

---

## 🚨 What to Watch For

### ✅ Good Signs:

- Loss steadily decreasing
- GPU utilization 90-100%
- Temperature stable (40-70°C)
- Memory usage stable
- Learning rate decreasing smoothly

### ⚠️ Warning Signs:

- Loss stuck at same value
- Loss increasing
- GPU utilization < 50%
- Temperature > 80°C
- Memory steadily increasing (potential leak)

### ❌ Problem Signs:

- Loss = NaN
- CUDA Out of Memory
- Temperature > 90°C
- GPU utilization = 0%

---

## 📊 Post-Training Analysis

### View TensorBoard Logs:

```bash
tensorboard --logdir checkpoints/ --port 6006
```

### Analyze GPU Logs:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load GPU stats
df = pd.read_csv('gpu_stats.csv')

# Plot memory usage over time
plt.plot(df['timestamp'], df['memory_usage_percent'])
plt.xlabel('Time')
plt.ylabel('GPU Memory %')
plt.title('GPU Memory Usage During Training')
plt.show()
```

### Parse Training Logs:

```python
import re

losses = []
with open('training.log', 'r') as f:
    for line in f:
        match = re.search(r"'loss':\s*([0-9.]+)", line)
        if match:
            losses.append(float(match.group(1)))

# Plot loss curve
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
```

---

## 🎓 Understanding the Metrics

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed explanations of:
- What is Loss?
- What is Learning Rate?
- What is an Epoch?
- How to interpret GPU stats?

---

**Happy Monitoring! 📊🚀**
