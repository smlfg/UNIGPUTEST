#!/usr/bin/env python3
"""
Mistral 7B Test - Testing larger model with quantization
Demonstrates FP16, 8-bit, and 4-bit loading
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc

def test_mistral(quantization=None):
    """
    Test Mistral 7B with different quantization

    Args:
        quantization: None (FP16), "8bit", or "4bit"
    """
    model_name = "mistralai/Mistral-7B-v0.1"

    print(f"\n{'='*70}")
    print(f"  Testing Mistral 7B - {quantization if quantization else 'FP16'}")
    print(f"{'='*70}\n")

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Get starting memory
    start_mem = torch.cuda.memory_allocated(0) / 1e9

    # Load kwargs
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16
    }

    if quantization == "8bit":
        load_kwargs["load_in_8bit"] = True
    elif quantization == "4bit":
        load_kwargs["load_in_4bit"] = True

    try:
        # Load model
        print("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"📦 Loading model ({quantization if quantization else 'FP16'})...")
        start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        load_time = time.time() - start

        # Memory after loading
        loaded_mem = torch.cuda.memory_allocated(0) / 1e9
        mem_used = loaded_mem - start_mem

        print(f"✅ Model loaded in {load_time:.2f}s")
        print(f"💾 GPU Memory: {mem_used:.2f} GB\n")

        # Test generation
        print("🤖 Testing generation...\n")

        prompt = "Explain quantum computing in simple terms:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        gen_time = time.time() - start

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_per_sec = 100 / gen_time

        print(f"Generated text:\n{text}\n")
        print(f"⚡ Speed: {tokens_per_sec:.1f} tokens/sec")
        print(f"⏱️  Time: {gen_time:.2f}s")

        # Cleanup
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        return {
            'success': True,
            'load_time': load_time,
            'memory_gb': mem_used,
            'tokens_per_sec': tokens_per_sec,
            'gen_time': gen_time
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return {'success': False, 'error': str(e)}

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         Mistral 7B Quantization Test Suite                      ║
    ║         NVIDIA L40S (48GB VRAM)                                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}\n")

    # Test different quantizations
    tests = [
        # Start with 4-bit (most memory efficient)
        "4bit",
        # Then 8-bit
        "8bit",
        # Finally FP16 (if you have HF token and want to test)
        # None,  # Uncomment to test FP16
    ]

    results = []

    for quant in tests:
        result = test_mistral(quant)
        results.append({
            'quant': quant if quant else 'FP16',
            **result
        })
        time.sleep(2)  # Brief pause between tests

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Quantization':<15} {'Load Time':<12} {'Memory':<12} {'Tokens/s':<12} {'Status'}")
    print("-" * 70)

    for r in results:
        if r['success']:
            print(f"{r['quant']:<15} {r['load_time']:<12.2f} "
                  f"{r['memory_gb']:<12.2f} {r['tokens_per_sec']:<12.1f} ✅")
        else:
            print(f"{r['quant']:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} ❌")

    print(f"\n{'='*70}")
    print("Testing complete!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
