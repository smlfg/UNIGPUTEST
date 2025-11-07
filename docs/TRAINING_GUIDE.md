# 🎓 Training Guide - Understanding LLM Fine-Tuning

## 📚 Table of Contents

1. [What is Fine-Tuning?](#what-is-fine-tuning)
2. [Understanding the Metrics](#understanding-the-metrics)
3. [What Happens During Training](#what-happens-during-training)
4. [Monitoring Your Training](#monitoring-your-training)
5. [Troubleshooting](#troubleshooting)

---

## 🎯 What is Fine-Tuning?

**Fine-tuning** is teaching a pre-trained AI model to be better at a specific task.

### Analogy:
```
Pre-trained Model = University graduate (knows general knowledge)
Fine-tuning       = Specialized training (becomes expert in Python coding)
Result            = Expert Python programmer
```

### What We're Doing:
- **Model**: Llama 3.2-1B (1 billion parameters)
- **Task**: Teaching it to write Python code
- **Dataset**: 18,000 Python code examples
- **Method**: QLoRA (efficient fine-tuning)

---

## 📊 Understanding the Metrics

### 1. **Loss** (Most Important!)

**What it is:**
- A "wrongness score"
- **Lower = Better!**
- Measures how far model predictions are from correct answers

**Typical Values:**
```
2.5-3.0  ← Very bad (start of training)
1.5-2.0  ← Getting better
1.0-1.5  ← Good progress
0.5-1.0  ← Very good!
0.3-0.5  ← Excellent! (well-trained)
< 0.3    ← Outstanding! (might be overfitting)
```

**Example:**
```python
Prompt: "Write function to add two numbers"

Model predicts: "def add(x, y):"
Correct answer: "def add(a, b):"
                      ❌    ❌
Loss = 0.8 (pretty good, just wrong variable names)

Model predicts: "def add(a, b):"
Correct answer: "def add(a, b):"
                      ✅    ✅
Loss = 0.1 (excellent!)
```

### 2. **Learning Rate**

**What it is:**
- How big the "steps" are when updating the model
- Too large = model jumps around, can't settle
- Too small = learning is very slow

**Our Strategy (Cosine Schedule):**
```
Start:  0.0002  ← Large steps (explore quickly)
Middle: 0.00015 ← Medium steps (refine)
End:    0.00005 ← Small steps (fine-tune precisely)
```

**Analogy:**
```
Finding a parking spot:
├─ Far away → Drive fast (large LR)
├─ Getting close → Slow down (medium LR)
└─ Almost there → Crawl slowly (small LR) ← Perfect parking!
```

### 3. **Epoch**

**What it is:**
- One complete pass through the entire dataset
- Epoch 0.5 = Seen 50% of data
- Epoch 1.0 = Seen 100% of data (all 18,000 examples once)
- Epoch 3.0 = Seen all data 3 times

**Why Multiple Epochs?**
```
Epoch 1: Learn basics (syntax, structure)
Epoch 2: Recognize patterns (common solutions)
Epoch 3: Refine & optimize (edge cases, details)
```

---

## 🔬 What Happens During Training

### The Training Loop (Repeated ~4,000 times)

```python
for each batch of 4 examples:

    # 1. FORWARD PASS - Make predictions
    predictions = model(batch)
    # Model tries to generate code

    # 2. CALCULATE LOSS - How wrong are we?
    loss = compare(predictions, correct_answers)
    # e.g., loss = 1.234

    # 3. BACKWARD PASS - Calculate gradients
    gradients = loss.backward()
    # "If I change this parameter by +0.001, loss decreases by 0.05"

    # 4. UPDATE PARAMETERS - Adjust LoRA weights
    for param in lora_adapter:
        param = param - (learning_rate × gradient)
    # Each parameter gets slightly better

    # 5. REPEAT with next batch!
```

### What the Model Learns:

**After 100 steps (Epoch 0.1):**
```python
Prompt: "reverse a string"
Output: "I think we can use Python to..." ❌
# Still confused
```

**After 1000 steps (Epoch 1.0):**
```python
Prompt: "reverse a string"
Output: "def reverse(s):\n    return s[::-1]" ✅
# Basic syntax learned!
```

**After 3000 steps (Epoch 3.0):**
```python
Prompt: "reverse a string with error handling"
Output: "def reverse(s):\n    if not isinstance(s, str):\n        raise TypeError\n    return s[::-1]" ✅✅
# Advanced patterns learned!
```

---

## 📺 Monitoring Your Training

### Method 1: Watch Training Output

```bash
python src/training/train.py
```

**What you'll see:**
```
🔥 Starting training...
{'loss': 2.451, 'learning_rate': 0.0002, 'epoch': 0.01}  ← Step 10
{'loss': 2.387, 'learning_rate': 0.00019998, 'epoch': 0.02}  ← Step 20
{'loss': 2.301, 'learning_rate': 0.00019995, 'epoch': 0.03}  ← Step 30
...
```

**Watch for:**
- ✅ Loss **decreasing** steadily
- ✅ Epoch **increasing** gradually
- ❌ Loss **increasing** (something wrong!)
- ❌ Loss **stuck** at same value (learning stalled)

### Method 2: GPU Monitoring

**In separate terminal:**
```bash
python src/utils/gpu_monitor.py
```

**Shows:**
```
🖥️  GPU MONITORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💾 VRAM: [████████████░░░░] 67%
   Used: 6.8 GB / 44.4 GB

⚡ GPU Util: [█████████████████░] 98%

🌡️  Temp: 🟢 45°C
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Healthy training:**
- GPU Util: 90-100% (GPU is busy!)
- VRAM: 6-10 GB (model + data loaded)
- Temp: < 80°C (cool and stable)

### Method 3: Training Dashboard

**In separate terminal:**
```bash
# First redirect training output to file:
python src/training/train.py > training.log 2>&1

# Then monitor in another terminal:
python src/utils/training_dashboard.py --log training.log
```

**Shows:**
- 📊 Current progress
- 📈 Loss trends
- ⏱️ Time estimates
- 📉 Statistics

### Method 4: TensorBoard (Advanced)

```bash
# Start TensorBoard:
tensorboard --logdir checkpoints/logs --port 6006

# Open in browser:
http://localhost:6006
```

**Features:**
- Beautiful graphs of loss over time
- Learning rate schedule visualization
- Gradient flow analysis
- Model architecture viewer

---

## ⏱️ Timeline - What to Expect

```
MINUTE 0-5: Setup
├─ Loading model (2.5 GB download)
├─ Loading dataset
├─ Initializing training
└─ Status: Preparing...

MINUTE 5-15: Early Training (Epoch 0.0 - 0.2)
├─ Loss: 2.5 → 2.0
├─ Learning syntax
├─ GPU: 98% utilized
└─ Status: Learning fast!

MINUTE 15-40: Mid Training (Epoch 0.2 - 0.8)
├─ Loss: 2.0 → 1.2
├─ Learning patterns
├─ Saving checkpoints
└─ Status: Steady progress

MINUTE 40-60: Epoch 1 Complete (Epoch 0.8 - 1.0)
├─ Loss: 1.2 → 0.9
├─ First pass through data done
├─ Model can write basic code
└─ Status: First epoch done! ✓

MINUTE 60-90: Epoch 2 (Epoch 1.0 - 2.0)
├─ Loss: 0.9 → 0.6
├─ Refining knowledge
├─ Better code quality
└─ Status: Getting good!

MINUTE 90-120: Epoch 3 (Epoch 2.0 - 3.0)
├─ Loss: 0.6 → 0.4
├─ Final optimizations
├─ Handling edge cases
└─ Status: Almost done!

MINUTE 120: Finished! 🎉
├─ Final loss: ~0.4
├─ Model saved
├─ Ready to use!
└─ Status: Complete!
```

---

## 🐛 Troubleshooting

### Problem: Loss Not Decreasing

**Symptoms:**
```
{'loss': 2.451, 'epoch': 0.1}
{'loss': 2.447, 'epoch': 0.2}  ← Barely changing
{'loss': 2.445, 'epoch': 0.3}
```

**Solutions:**
1. ✅ **Wait longer** - Needs 50-100 steps to see progress
2. ✅ **Check learning rate** - Might be too small
3. ✅ **Check dataset** - Is it loading correctly?

### Problem: Loss Increasing

**Symptoms:**
```
{'loss': 1.234, 'epoch': 0.5}
{'loss': 1.456, 'epoch': 0.6}  ❌ Going up!
{'loss': 1.789, 'epoch': 0.7}
```

**Solutions:**
1. ❌ **Stop training** - Something is wrong
2. ✅ **Lower learning rate** - Reduce to 1e-5
3. ✅ **Check GPU memory** - Might be OOM

### Problem: Training Crashed

**Error:**
```
CUDA out of memory
```

**Solutions:**
```bash
# Edit config to use less memory:
nano configs/training_config.yaml

# Reduce batch size:
per_device_train_batch_size: 2  # Was 4

# Reduce sequence length:
max_seq_length: 1024  # Was 2048
```

### Problem: Very Slow Training

**GPU Util < 50%:**

**Possible causes:**
1. ❌ Dataset loading bottleneck
2. ❌ Too small batch size
3. ❌ CPU bottleneck

**Solutions:**
```bash
# Increase batch size:
per_device_train_batch_size: 8  # Was 4

# Use more workers:
dataloader_num_workers: 4
```

---

## 🎯 After Training Completes

### Test Your Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Load your fine-tuned LoRA adapter
model = PeftModel.from_pretrained(model, "checkpoints/final_model")

# Test it!
prompt = "Write a Python function to calculate fibonacci numbers"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0]))
```

### Compare Before/After

```python
# Before fine-tuning (base model):
"I can help with that. Fibonacci numbers are..."

# After fine-tuning (your model):
"def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"
```

---

## 📚 Key Takeaways

| Concept | What You Learned |
|---------|------------------|
| **Loss** | Lower is better - measures wrongness |
| **Learning Rate** | Step size - decreases over time |
| **Epoch** | Pass through dataset - we do 3 |
| **LoRA** | Only train 1.5% of model - efficient! |
| **Gradients** | Tell us how to improve parameters |
| **Checkpoints** | Saved models - can resume if crash |

---

## 🎓 Educational Value

**What makes this framework special:**

1. ✅ **Fully Documented** - Every concept explained
2. ✅ **Real-time Monitoring** - See what's happening
3. ✅ **Production-Ready** - Actually works on real GPUs
4. ✅ **Educational** - Understand WHY, not just HOW
5. ✅ **Extensible** - Easy to modify for your needs

**Perfect for:**
- 🎓 Bachelor/Master Thesis
- 💼 Portfolio Projects
- 📚 Learning ML Engineering
- 🏢 Production Deployments

---

**Questions? Check the main README or open an issue on GitHub!** 🚀
