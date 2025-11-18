#!/usr/bin/env python3
"""
Advanced LLM Testing Suite
===========================

Testet state-of-the-art Open-Source Modelle auf der L40S:
- Mistral 7B (SOTA open-source)
- Phi-3 Mini (Microsoft, sehr effizient)
- Gemma 2B (Google)
- Optional: Llama 2 7B (braucht HF Token)

Fokus auf 4-bit Quantisierung (beste Performance/Memory Balance)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelTestResult:
    """Results für einen Model-Test"""
    model_name: str
    load_time_sec: float
    memory_gb: float
    first_token_latency_ms: float
    throughput_tokens_per_sec: float
    sample_output: str
    success: bool
    error: Optional[str] = None


def print_header(title: str):
    """Schöner Header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def cleanup_gpu():
    """GPU Memory aufräumen"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_model(
    model_id: str,
    display_name: str,
    prompt: str = "Explain quantum computing in simple terms:",
    num_tokens: int = 100,
    use_4bit: bool = True
) -> ModelTestResult:
    """
    Testet ein einzelnes Modell

    Args:
        model_id: HuggingFace Model ID
        display_name: Name für Ausgabe
        prompt: Test-Prompt
        num_tokens: Tokens zum Generieren
        use_4bit: 4-bit Quantisierung verwenden
    """
    print_header(f"{display_name}")

    cleanup_gpu()
    start_mem = torch.cuda.memory_allocated(0) / 1e9

    try:
        # 1. LOADING
        print(f"📦 Loading {display_name}...")
        if use_4bit:
            print("   Using 4-bit quantization (optimal for L40S)")

        load_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16
        }

        if use_4bit:
            load_kwargs["load_in_4bit"] = True

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs
        )

        load_time = time.time() - load_start

        # 2. MEMORY
        torch.cuda.synchronize()
        memory_gb = (torch.cuda.memory_allocated(0) / 1e9) - start_mem

        print(f"✅ Loaded in {load_time:.1f}s")
        print(f"💾 GPU Memory: {memory_gb:.2f} GB")

        # 3. FIRST TOKEN LATENCY
        print(f"\n🔥 Testing latency...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        torch.cuda.synchronize()
        ftl_start = time.time()

        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

        torch.cuda.synchronize()
        ftl_ms = (time.time() - ftl_start) * 1000

        print(f"⚡ First Token Latency: {ftl_ms:.1f}ms")

        # 4. WARMUP
        print(f"\n🚀 Warming up...")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
        torch.cuda.synchronize()

        # 5. THROUGHPUT TEST
        print(f"🚀 Testing throughput ({num_tokens} tokens)...")

        throughput_start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1  # Verhindert Wiederholungen
            )

        torch.cuda.synchronize()
        throughput_time = time.time() - throughput_start
        throughput = num_tokens / throughput_time

        # Generierter Text
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n✅ Throughput: {throughput:.1f} tokens/sec")
        print(f"⏱️  Time: {throughput_time:.1f}s")
        print(f"\n📝 Generated Output:")
        print(f"{'─'*80}")
        print(full_text)
        print(f"{'─'*80}\n")

        # Result
        result = ModelTestResult(
            model_name=display_name,
            load_time_sec=load_time,
            memory_gb=memory_gb,
            first_token_latency_ms=ftl_ms,
            throughput_tokens_per_sec=throughput,
            sample_output=full_text,
            success=True
        )

        # Cleanup
        del model, tokenizer
        cleanup_gpu()

        print(f"✅ {display_name} test completed!\n")

        return result

    except Exception as e:
        print(f"\n❌ Error testing {display_name}: {str(e)}\n")
        cleanup_gpu()

        return ModelTestResult(
            model_name=display_name,
            load_time_sec=0,
            memory_gb=0,
            first_token_latency_ms=0,
            throughput_tokens_per_sec=0,
            sample_output="",
            success=False,
            error=str(e)
        )


def main():
    """Hauptprogramm"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║          Advanced LLM Testing Suite - NVIDIA L40S                   ║
    ║          State-of-the-Art Open-Source Models                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # GPU Check
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}\n")

    # Test-Prompt (konsistent für alle Modelle)
    test_prompt = "Explain quantum computing in simple terms:"

    # Modelle zum Testen
    models = [
        # Klein & effizient (gut zum Starten)
        ("microsoft/phi-2", "Phi-2 (2.7B)", 100),

        # Google's kompaktes Modell
        ("google/gemma-2b", "Gemma 2B", 100),

        # State-of-the-art (empfohlen!)
        ("mistralai/Mistral-7B-v0.1", "Mistral 7B", 150),

        # Optional: Llama 2 (braucht HuggingFace Token!)
        # Kommentiere ein, wenn du Token hast:
        # ("meta-llama/Llama-2-7b-chat-hf", "Llama 2 7B Chat", 150),
    ]

    results = []

    # Teste alle Modelle
    for i, (model_id, display_name, num_tokens) in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"  TEST {i}/{len(models)}")
        print(f"{'='*80}")

        result = test_model(
            model_id=model_id,
            display_name=display_name,
            prompt=test_prompt,
            num_tokens=num_tokens,
            use_4bit=True  # Immer 4-bit für beste Performance
        )

        results.append(result)

        # Kurze Pause zwischen Tests
        time.sleep(2)

    # ZUSAMMENFASSUNG
    print_header("RESULTS SUMMARY")

    print(f"{'Model':<20} {'Load(s)':<10} {'Memory(GB)':<12} {'FTL(ms)':<10} "
          f"{'Throughput':<15} {'Status'}")
    print("─" * 90)

    for r in results:
        if r.success:
            print(f"{r.model_name:<20} {r.load_time_sec:<10.1f} "
                  f"{r.memory_gb:<12.2f} {r.first_token_latency_ms:<10.1f} "
                  f"{r.throughput_tokens_per_sec:<15.1f} ✅")
        else:
            print(f"{r.model_name:<20} {'N/A':<10} {'N/A':<12} {'N/A':<10} "
                  f"{'N/A':<15} ❌ {r.error[:30]}")

    print("\n" + "="*90)

    # Beste Performance
    successful = [r for r in results if r.success]
    if successful:
        fastest = max(successful, key=lambda x: x.throughput_tokens_per_sec)
        smallest = min(successful, key=lambda x: x.memory_gb)

        print(f"\n🏆 Fastest Model: {fastest.model_name} "
              f"({fastest.throughput_tokens_per_sec:.1f} tok/s)")
        print(f"💾 Most Efficient: {smallest.model_name} "
              f"({smallest.memory_gb:.2f} GB)")

    print("\n" + "="*90)
    print("✅ All tests completed!")
    print("="*90 + "\n")

    # Empfehlungen
    print_header("RECOMMENDATIONS")

    print("""
    📊 Based on your L40S (48GB VRAM):

    🎯 For Production Use:
       - Mistral 7B (4-bit): Best quality/performance balance
       - Can run 12-14 instances simultaneously!

    ⚡ For Speed-Critical Apps:
       - Phi-2: Fastest, smallest, great for simple tasks
       - Gemma 2B: Good quality, very fast

    💰 For Cost Optimization:
       - 4-bit quantization is the sweet spot
       - Minimal quality loss, 4x memory savings

    🚀 Next Steps:
       - Try batch inference (higher throughput)
       - Test with your own prompts
       - Fine-tune with LoRA/QLoRA
    """)


if __name__ == "__main__":
    main()
