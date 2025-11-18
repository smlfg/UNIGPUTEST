#!/usr/bin/env python3
"""
LLM Benchmark Suite für NVIDIA L40S
====================================

Dieses Script vergleicht verschiedene LLMs und Quantisierungsmethoden.

METRIKEN:
---------
1. Load Time: Zeit zum Laden des Modells (Sekunden)
   - Wichtig für: Cold-Start-Szenarien, Serverless

2. GPU Memory: VRAM-Verbrauch (GB)
   - Wichtig für: Multi-Model-Serving, Batch-Size-Planung

3. First Token Latency: Zeit bis zum ersten generierten Token (ms)
   - Wichtig für: User Experience, Chat-Anwendungen

4. Throughput: Tokens pro Sekunde
   - Wichtig für: Batch-Processing, Gesamtperformance

5. Memory Bandwidth: Datendurchsatz GPU ↔ Memory (GB/s)
   - Wichtig für: Verständnis von Bottlenecks

6. Tokens per Watt: Effizienz-Metrik (tokens/sec/W)
   - Wichtig für: Kosten, Nachhaltigkeit

QUANTISIERUNG:
--------------
- FP16 (16-bit): Volle Präzision, höchste Qualität
  * ~2x Speicher von FP32
  * Gute Balance zwischen Speed und Qualität

- INT8 (8-bit): Mittlere Kompression
  * ~50% Speicher von FP16
  * Minimaler Qualitätsverlust (<1%)
  * 1.5-2x schneller als FP16

- INT4 (4-bit): Maximale Kompression
  * ~25% Speicher von FP16
  * Geringer Qualitätsverlust (1-3%)
  * Bis zu 4x schneller als FP16

MODELLE ZUM TESTEN:
------------------
- GPT-2 Small (124M): Baseline, schnell
- GPT-2 Medium (355M): Mittlere Größe
- GPT-2 Large (774M): Größer, bessere Qualität
- Mistral 7B (7B): State-of-the-art Open-Source
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import gc
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import sys

@dataclass
class BenchmarkResult:
    """Speichert alle Benchmark-Metriken für einen Test"""
    model_name: str
    quantization: str
    load_time_sec: float
    memory_gb: float
    first_token_latency_ms: float
    throughput_tokens_per_sec: float
    total_inference_time_sec: float
    num_tokens_generated: int
    success: bool
    error_msg: Optional[str] = None

    def memory_reduction_vs_fp16(self, fp16_memory: float) -> float:
        """Berechnet Speicherersparnis gegenüber FP16"""
        if fp16_memory == 0:
            return 0
        return ((fp16_memory - self.memory_gb) / fp16_memory) * 100

    def speedup_vs_fp16(self, fp16_throughput: float) -> float:
        """Berechnet Speedup gegenüber FP16"""
        if fp16_throughput == 0:
            return 0
        return self.throughput_tokens_per_sec / fp16_throughput


class LLMBenchmark:
    """Benchmark-Framework für LLM Testing"""

    def __init__(self, output_json: str = "benchmark_results.json"):
        self.results: List[BenchmarkResult] = []
        self.output_json = output_json

    def print_section(self, title: str):
        """Formatierte Section-Header"""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")

    def cleanup_memory(self):
        """Räumt GPU Memory auf"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def benchmark_model(
        self,
        model_name: str,
        quantization: str,
        load_kwargs: Dict,
        test_prompt: str = "The future of artificial intelligence is",
        num_tokens: int = 50
    ) -> BenchmarkResult:
        """
        Führt einen kompletten Benchmark durch

        Args:
            model_name: HuggingFace Model ID
            quantization: "FP16", "8-bit", oder "4-bit"
            load_kwargs: Parameter für model.from_pretrained()
            test_prompt: Text für Generation
            num_tokens: Anzahl zu generierender Tokens

        Returns:
            BenchmarkResult mit allen Metriken
        """
        self.print_section(f"{model_name} - {quantization}")

        # Memory cleanup vor dem Test
        self.cleanup_memory()
        start_memory = torch.cuda.memory_allocated(0) / 1e9

        try:
            # 1. MODEL LOADING
            print(f"📦 Loading model...")
            load_start = time.time()

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )

            load_time = time.time() - load_start

            # 2. MEMORY USAGE
            torch.cuda.synchronize()
            loaded_memory = torch.cuda.memory_allocated(0) / 1e9
            memory_used = loaded_memory - start_memory

            print(f"✅ Loaded in {load_time:.2f}s")
            print(f"💾 GPU Memory: {memory_used:.3f} GB")

            # 3. FIRST TOKEN LATENCY
            print(f"\n🔥 Testing first token latency...")
            inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

            torch.cuda.synchronize()
            first_token_start = time.time()

            with torch.no_grad():
                # Generate nur 1 Token für Latency-Test
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False  # Greedy für konsistente Messungen
                )

            torch.cuda.synchronize()
            first_token_latency = (time.time() - first_token_start) * 1000  # ms

            print(f"⚡ First Token Latency: {first_token_latency:.2f}ms")

            # 4. THROUGHPUT TEST
            print(f"\n🚀 Testing throughput ({num_tokens} tokens)...")

            # Warmup (wichtig für faire Messungen!)
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )

            torch.cuda.synchronize()

            # Eigentliche Messung
            throughput_start = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=num_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9
                )

            torch.cuda.synchronize()
            throughput_time = time.time() - throughput_start
            throughput = num_tokens / throughput_time

            # Generierter Text (für Qualitätsprüfung)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"✅ Throughput: {throughput:.1f} tokens/sec")
            print(f"⏱️  Total time: {throughput_time:.2f}s")
            print(f"\n📝 Generated text preview:")
            print(f"   {generated_text[:150]}...")

            # Result erstellen
            result = BenchmarkResult(
                model_name=model_name,
                quantization=quantization,
                load_time_sec=load_time,
                memory_gb=memory_used,
                first_token_latency_ms=first_token_latency,
                throughput_tokens_per_sec=throughput,
                total_inference_time_sec=throughput_time,
                num_tokens_generated=num_tokens,
                success=True
            )

            # Cleanup
            del model
            del tokenizer
            self.cleanup_memory()

            print(f"\n✅ Benchmark completed successfully!")

            return result

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

            self.cleanup_memory()

            return BenchmarkResult(
                model_name=model_name,
                quantization=quantization,
                load_time_sec=0,
                memory_gb=0,
                first_token_latency_ms=0,
                throughput_tokens_per_sec=0,
                total_inference_time_sec=0,
                num_tokens_generated=0,
                success=False,
                error_msg=str(e)
            )

    def add_result(self, result: BenchmarkResult):
        """Fügt Result zur Liste hinzu"""
        self.results.append(result)

    def print_summary(self):
        """Zeigt Zusammenfassung aller Tests"""
        self.print_section("BENCHMARK SUMMARY")

        if not self.results:
            print("No results to display.")
            return

        # Erfolgreiche Tests filtern
        successful = [r for r in self.results if r.success]

        if not successful:
            print("❌ All benchmarks failed!")
            return

        # Header
        print(f"{'Model':<20} {'Quant':<8} {'Load(s)':<10} {'Mem(GB)':<10} "
              f"{'FTL(ms)':<10} {'Throughput':<12} {'Status'}")
        print("-" * 90)

        # Results
        for r in self.results:
            if r.success:
                status = "✅"
                print(f"{r.model_name:<20} {r.quantization:<8} "
                      f"{r.load_time_sec:<10.2f} {r.memory_gb:<10.3f} "
                      f"{r.first_token_latency_ms:<10.2f} "
                      f"{r.throughput_tokens_per_sec:<12.1f} {status}")
            else:
                status = f"❌ {r.error_msg[:30]}"
                print(f"{r.model_name:<20} {r.quantization:<8} "
                      f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {status}")

        print("\n" + "="*90)

        # Vergleichsanalyse
        self.print_section("QUANTIZATION COMPARISON")

        # Gruppiere nach Model
        by_model = {}
        for r in successful:
            if r.model_name not in by_model:
                by_model[r.model_name] = {}
            by_model[r.model_name][r.quantization] = r

        for model_name, quants in by_model.items():
            if 'FP16' not in quants:
                continue

            fp16 = quants['FP16']
            print(f"\n📊 {model_name} Comparison (vs FP16 baseline):\n")
            print(f"{'Quantization':<12} {'Memory Saved':<15} {'Speedup':<12} {'FTL Change'}")
            print("-" * 55)

            for quant_name, result in quants.items():
                mem_saved = result.memory_reduction_vs_fp16(fp16.memory_gb)
                speedup = result.speedup_vs_fp16(fp16.throughput_tokens_per_sec)
                ftl_change = ((result.first_token_latency_ms - fp16.first_token_latency_ms)
                             / fp16.first_token_latency_ms * 100)

                print(f"{quant_name:<12} {mem_saved:>6.1f}%{'':<8} "
                      f"{speedup:>5.2f}x{'':<5} {ftl_change:>+6.1f}%")

        print()

    def save_results(self):
        """Speichert Results als JSON"""
        data = {
            'results': [asdict(r) for r in self.results],
            'metadata': {
                'gpu': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
                'total_vram_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        }

        with open(self.output_json, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Results saved to: {self.output_json}")


def main():
    """Hauptprogramm"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                  LLM Benchmark Suite - NVIDIA L40S                   ║
    ║              Professionelle Metrik-basierte Model-Evaluierung        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # GPU Check
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Benchmark initialisieren
    benchmark = LLMBenchmark(output_json="llm_benchmark_results.json")

    # Test-Konfiguration
    # Format: (model_name, display_name_for_model)
    models_to_test = [
        ("gpt2", "GPT-2 (124M)"),
        ("gpt2-medium", "GPT-2-Medium (355M)"),
        # ("gpt2-large", "GPT-2-Large (774M)"),  # Optional: größeres Modell
    ]

    # Quantisierungen zum Testen
    quantizations = [
        ("FP16", {"torch_dtype": torch.float16, "device_map": "auto"}),
        ("8-bit", {"load_in_8bit": True, "device_map": "auto"}),
        ("4-bit", {"load_in_4bit": True, "device_map": "auto"}),
    ]

    test_prompt = "The future of artificial intelligence is"
    num_tokens = 50

    # Durchlaufe alle Kombinationen
    total_tests = len(models_to_test) * len(quantizations)
    current_test = 0

    for model_id, model_display in models_to_test:
        for quant_name, quant_kwargs in quantizations:
            current_test += 1
            print(f"\n{'='*80}")
            print(f"  TEST {current_test}/{total_tests}")
            print(f"{'='*80}")

            result = benchmark.benchmark_model(
                model_name=model_id,
                quantization=quant_name,
                load_kwargs=quant_kwargs,
                test_prompt=test_prompt,
                num_tokens=num_tokens
            )

            benchmark.add_result(result)

            # Kurze Pause zwischen Tests
            time.sleep(2)

    # Zusammenfassung und Speicherung
    benchmark.print_summary()
    benchmark.save_results()

    print("\n" + "="*80)
    print("  ✅ ALL BENCHMARKS COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
