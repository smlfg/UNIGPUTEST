#!/usr/bin/env python3
"""
Quick GPT-2 Test - First LLM on L40S
Tests basic loading and inference
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def main():
    print("\n" + "="*70)
    print("  Quick GPT-2 Test - NVIDIA L40S")
    print("="*70 + "\n")

    # GPU Check
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Load model
    print("📦 Loading GPT-2...")
    model_name = "gpt2"

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    load_time = time.time() - start

    print(f"✅ Model loaded in {load_time:.2f}s")

    # Memory check
    mem_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"💾 GPU Memory used: {mem_used:.2f} GB\n")

    # Test generation
    print("🤖 Generating text...\n")

    prompts = [
        "The future of AI is",
        "Machine learning will",
        "Neural networks are"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}/3 ---")
        print(f"Prompt: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        gen_time = time.time() - start

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_per_sec = 30 / gen_time

        print(f"Output: {text}")
        print(f"⚡ Speed: {tokens_per_sec:.1f} tokens/sec ({gen_time:.2f}s)")

    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
