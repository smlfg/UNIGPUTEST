#!/usr/bin/env python3
"""
LLM Loading and Inference Test
Tests different quantization methods on NVIDIA L40S
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return allocated, reserved, total
    return 0, 0, 0

def test_model_loading(model_name="gpt2", quantization=None):
    """
    Test loading a model with different quantization settings

    Args:
        model_name: HuggingFace model name
        quantization: None, "8bit", or "4bit"
    """
    print_section(f"Testing: {model_name} ({quantization if quantization else 'FP16'})")

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    start_mem_alloc, start_mem_res, total_mem = get_gpu_memory()
    print(f"Starting GPU Memory:")
    print(f"  Allocated: {start_mem_alloc:.2f} GB")
    print(f"  Reserved:  {start_mem_res:.2f} GB")
    print(f"  Total:     {total_mem:.2f} GB\n")

    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare load kwargs
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16
        }

        if quantization == "8bit":
            load_kwargs["load_in_8bit"] = True
        elif quantization == "4bit":
            load_kwargs["load_in_4bit"] = True

        # Load model
        print(f"Loading model with {quantization if quantization else 'FP16'}...")
        start_time = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        load_time = time.time() - start_time

        # Get memory after loading
        load_mem_alloc, load_mem_res, _ = get_gpu_memory()

        print(f"\n✓ Model loaded successfully!")
        print(f"  Load time: {load_time:.2f}s")
        print(f"  GPU Memory after loading:")
        print(f"    Allocated: {load_mem_alloc:.2f} GB (+{load_mem_alloc - start_mem_alloc:.2f} GB)")
        print(f"    Reserved:  {load_mem_res:.2f} GB (+{load_mem_res - start_mem_res:.2f} GB)")

        # Test inference
        print("\nTesting inference...")
        prompt = "The future of artificial intelligence is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        inference_time = time.time() - start_time

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n✓ Inference completed!")
        print(f"  Time: {inference_time:.2f}s")
        print(f"  Tokens/sec: {50/inference_time:.2f}")
        print(f"\nGenerated text:")
        print(f"  {generated_text}")

        # Final memory
        final_mem_alloc, final_mem_res, _ = get_gpu_memory()
        print(f"\nFinal GPU Memory:")
        print(f"  Allocated: {final_mem_alloc:.2f} GB")
        print(f"  Reserved:  {final_mem_res:.2f} GB")

        # Cleanup
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        return {
            'success': True,
            'load_time': load_time,
            'inference_time': inference_time,
            'memory_used': load_mem_alloc - start_mem_alloc,
            'tokens_per_sec': 50/inference_time
        }

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")

        # Cleanup on error
        gc.collect()
        torch.cuda.empty_cache()

        return {'success': False, 'error': str(e)}

def main():
    """Main test function"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         LLM Loading & Quantization Test Suite                   ║
    ║         NVIDIA L40S (48GB VRAM)                                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Test different models and quantizations
    tests = [
        # Start small
        ("gpt2", None),           # ~500MB FP16
        ("gpt2", "8bit"),         # ~250MB
        ("gpt2", "4bit"),         # ~125MB

        # Medium model (uncomment if you want to test)
        # ("gpt2-large", None),   # ~3GB FP16
        # ("gpt2-large", "8bit"), # ~1.5GB
        # ("gpt2-large", "4bit"), # ~750MB
    ]

    results = []

    for model_name, quantization in tests:
        result = test_model_loading(model_name, quantization)
        results.append({
            'model': model_name,
            'quantization': quantization if quantization else 'FP16',
            **result
        })
        time.sleep(2)  # Brief pause between tests

    # Summary
    print_section("Test Summary")

    print(f"{'Model':<15} {'Quant':<8} {'Load Time':<12} {'Memory':<12} {'Tokens/s':<12} {'Status'}")
    print("-" * 70)

    for r in results:
        if r['success']:
            print(f"{r['model']:<15} {r['quantization']:<8} "
                  f"{r['load_time']:<12.2f} {r['memory_used']:<12.2f} "
                  f"{r['tokens_per_sec']:<12.2f} ✓")
        else:
            print(f"{r['model']:<15} {r['quantization']:<8} {'N/A':<12} {'N/A':<12} "
                  f"{'N/A':<12} ✗ ({r['error'][:30]}...)")

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
