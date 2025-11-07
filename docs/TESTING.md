# 🧪 Model Testing Guide

## Overview

This guide explains how to test your fine-tuned model using the comprehensive test prompt collection.

---

## 📋 Test Prompt Collection

Located in `data/test_prompts.json`, contains **62 test prompts** across 5 categories:

### Categories:

1. **Basic (10 prompts)** - Easy
   - Functions, strings, lists, loops
   - Perfect for validating basic functionality

2. **Intermediate (12 prompts)** - Medium
   - Algorithms, data structures, recursion
   - Tests deeper understanding

3. **Advanced (10 prompts)** - Hard
   - Graph algorithms, dynamic programming
   - Tests complex problem-solving

4. **Specialized (10 prompts)** - Medium/Hard
   - Decorators, generators, async, type hints
   - Tests Python-specific features

5. **Real-World (10 prompts)** - Medium/Hard
   - API design, database queries, caching
   - Tests practical application skills

---

## 🚀 Quick Start

### Test All Categories (5 prompts)

```bash
python src/evaluation/test_model.py \
    --model checkpoints/final_model \
    --limit 5
```

### Test Basic Category Only

```bash
python src/evaluation/test_model.py \
    --model checkpoints/final_model \
    --category basic
```

### Test Specific Category with Limit

```bash
python src/evaluation/test_model.py \
    --model checkpoints/final_model \
    --category intermediate \
    --limit 3
```

### Test with Custom Output File

```bash
python src/evaluation/test_model.py \
    --model checkpoints/final_model \
    --category advanced \
    --output results/my_test_results.json
```

---

## 📊 Command Line Options

```bash
python src/evaluation/test_model.py [OPTIONS]

Required:
  --model PATH          Path to LoRA adapter (e.g., checkpoints/final_model)

Optional:
  --base-model NAME     Base model name (default: meta-llama/Llama-3.2-1B)
  --prompts FILE        Test prompts JSON file (default: data/test_prompts.json)
  --category CATEGORY   Category to test (default: all)
                        Choices: all, basic, intermediate, advanced, specialized, real_world
  --limit N             Max prompts to test (default: no limit)
  --output FILE         Output JSON file (default: results/test_results_TIMESTAMP.json)
  --max-tokens N        Max tokens to generate (default: 256)
  --temperature FLOAT   Sampling temperature (default: 0.7)
```

---

## 📝 Example Test Prompts

### Basic
```
"Write a Python function to calculate the factorial of a number."
"Write a function to reverse a string."
"Write a function to check if a string is a palindrome."
```

### Intermediate
```
"Write a function to implement binary search on a sorted list."
"Write a recursive function to calculate the nth Fibonacci number."
"Write a class to implement a stack with push, pop, and peek methods."
```

### Advanced
```
"Write a function to implement Dijkstra's shortest path algorithm."
"Write a function to solve the N-Queens problem using backtracking."
"Write a class to implement a Trie with insert, search, and prefix search."
```

### Real-World
```
"Write a function to parse server logs and extract IP addresses."
"Write a decorator that retries a function with exponential backoff."
"Write a class to implement a token bucket rate limiter."
```

---

## 📂 Output Format

Results are saved as JSON:

```json
{
  "timestamp": "2025-11-07T23:45:12.123456",
  "category": "basic",
  "limit": 5,
  "tests": [
    {
      "id": "basic_001",
      "category": "basic",
      "difficulty": "easy",
      "prompt": "Write a Python function to calculate the factorial of a number.",
      "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    }
  ]
}
```

---

## 🎯 Evaluation Criteria

When reviewing results, consider:

1. **Syntax** - Is the code syntactically correct?
2. **Functionality** - Does it solve the problem?
3. **Efficiency** - Is the solution efficient?
4. **Readability** - Is the code clean and well-structured?
5. **Best Practices** - Does it follow Python conventions?
6. **Error Handling** - Does it handle edge cases?

---

## 🔍 Comparing Before/After Fine-Tuning

### Test Base Model (Before Fine-Tuning)

```bash
python src/evaluation/test_model.py \
    --model meta-llama/Llama-3.2-1B \
    --category basic \
    --limit 3 \
    --output results/base_model_results.json
```

**Note:** For base model, skip the `--model` flag or point directly to base model.

### Test Fine-Tuned Model (After Fine-Tuning)

```bash
python src/evaluation/test_model.py \
    --model checkpoints/final_model \
    --category basic \
    --limit 3 \
    --output results/finetuned_model_results.json
```

### Compare Results

```python
import json

# Load both results
with open('results/base_model_results.json') as f:
    base_results = json.load(f)

with open('results/finetuned_model_results.json') as f:
    finetuned_results = json.load(f)

# Compare responses for same prompts
for base_test, ft_test in zip(base_results['tests'], finetuned_results['tests']):
    print(f"\nPrompt: {base_test['prompt']}")
    print(f"\nBase Model:\n{base_test['response']}")
    print(f"\nFine-Tuned Model:\n{ft_test['response']}")
    print("="*80)
```

---

## 📊 Sample Test Run

```bash
$ python src/evaluation/test_model.py --model checkpoints/final_model --category basic --limit 3

================================================================================
🧪 TESTING MODEL
================================================================================
Category: basic
Total prompts: 3
================================================================================

────────────────────────────────────────────────────────────────────────────────
Test 1/3
ID: basic_001
Category: basic | Difficulty: easy
────────────────────────────────────────────────────────────────────────────────
📝 Prompt: Write a Python function to calculate the factorial of a number.
────────────────────────────────────────────────────────────────────────────────
🤖 Response:
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
────────────────────────────────────────────────────────────────────────────────

[... more tests ...]

✅ Results saved to: results/test_results_basic_20251107_234512.json

================================================================================
📊 SUMMARY
================================================================================
Total tests run: 3
Category: basic
Results saved: results/test_results_basic_20251107_234512.json
================================================================================
```

---

## 💡 Tips

1. **Start with Basic** - Test basic prompts first to verify model works
2. **Use Limit** - Use `--limit` for quick testing during development
3. **Save Results** - Always save results for comparison and analysis
4. **Temperature** - Lower temperature (0.2-0.5) for more deterministic code
5. **Max Tokens** - Increase `--max-tokens` for complex algorithms
6. **Categories** - Test each category separately for focused evaluation

---

## 🎓 Educational Value

This testing framework helps you:

- **Understand fine-tuning impact** - See concrete before/after differences
- **Identify strengths/weaknesses** - Know what your model does well
- **Portfolio showcase** - Include test results in thesis/portfolio
- **Continuous improvement** - Track progress across training iterations

---

## 📁 Results Organization

Recommended structure:

```
results/
├── base_model/
│   ├── basic_20251107_120000.json
│   ├── intermediate_20251107_120100.json
│   └── advanced_20251107_120200.json
├── checkpoint_1000/
│   ├── basic_20251107_140000.json
│   └── ...
└── final_model/
    ├── basic_20251107_180000.json
    ├── intermediate_20251107_180100.json
    ├── advanced_20251107_180200.json
    ├── specialized_20251107_180300.json
    └── real_world_20251107_180400.json
```

---

## 🚨 Troubleshooting

### Out of Memory

```bash
# Reduce max tokens
python src/evaluation/test_model.py --model checkpoints/final_model --max-tokens 128

# Test fewer prompts at once
python src/evaluation/test_model.py --model checkpoints/final_model --limit 1
```

### Model Not Found

```bash
# Check if adapter exists
ls -la checkpoints/final_model/

# Should contain:
# - adapter_config.json
# - adapter_model.safetensors (or .bin)
```

### Poor Quality Responses

```bash
# Try different temperature
python src/evaluation/test_model.py --model checkpoints/final_model --temperature 0.3

# Try more tokens
python src/evaluation/test_model.py --model checkpoints/final_model --max-tokens 512
```

---

**Ready to test your model? Start with basic prompts and work your way up! 🚀**
