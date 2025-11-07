#!/usr/bin/env python3
"""
Benchmarking Suite
Measure latency, throughput, and memory usage for models
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    model_name: str
    model_type: str  # 'pytorch', 'onnx', etc.

    # Latency metrics (milliseconds)
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float

    # Throughput metrics
    tokens_per_second: float
    samples_per_second: float

    # Memory metrics (GB)
    peak_memory_gb: float
    avg_memory_gb: float

    # Input/output info
    num_runs: int
    input_length: int
    output_length: int
    batch_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def print_summary(self):
        """Print benchmark summary"""
        print("=" * 60)
        print(f"📊 BENCHMARK RESULTS: {self.model_name}")
        print("=" * 60)
        print(f"Model Type: {self.model_type}")
        print(f"Runs: {self.num_runs}")
        print(f"Input Length: {self.input_length} tokens")
        print(f"Output Length: {self.output_length} tokens")
        print(f"Batch Size: {self.batch_size}")
        print()
        print("⏱️  LATENCY:")
        print(f"   Mean:   {self.mean_latency_ms:.2f} ms")
        print(f"   Median: {self.median_latency_ms:.2f} ms")
        print(f"   P95:    {self.p95_latency_ms:.2f} ms")
        print(f"   P99:    {self.p99_latency_ms:.2f} ms")
        print(f"   Min:    {self.min_latency_ms:.2f} ms")
        print(f"   Max:    {self.max_latency_ms:.2f} ms")
        print()
        print("🚀 THROUGHPUT:")
        print(f"   Tokens/sec:  {self.tokens_per_second:.2f}")
        print(f"   Samples/sec: {self.samples_per_second:.2f}")
        print()
        print("💾 MEMORY:")
        print(f"   Peak: {self.peak_memory_gb:.2f} GB")
        print(f"   Avg:  {self.avg_memory_gb:.2f} GB")
        print("=" * 60)


class ModelBenchmark:
    """Benchmark models for performance metrics"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        warmup_runs: int = 5,
    ):
        """
        Initialize benchmarker

        Args:
            model_path: Path to model
            device: Device to run on ('cuda' or 'cpu')
            warmup_runs: Number of warmup runs before benchmarking
        """
        self.model_path = Path(model_path)
        self.device = device
        self.warmup_runs = warmup_runs

        self.model = None
        self.tokenizer = None

    def load_pytorch_model(self):
        """Load PyTorch model"""
        print(f"📦 Loading PyTorch model: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

        print("✅ Model loaded")

    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        num_runs: int = 100,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        """
        Run benchmark

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            num_runs: Number of benchmark runs
            batch_size: Batch size

        Returns:
            BenchmarkResult: Benchmark results
        """
        if self.model is None:
            self.load_pytorch_model()

        print(f"🏃 Running benchmark ({num_runs} runs)...")

        # Prepare inputs
        inputs = self.tokenizer(
            prompts[:batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        # Warmup
        print(f"   Warming up ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        # Clear GPU cache if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Benchmark runs
        latencies = []
        memory_usage = []

        print(f"   Benchmarking ({num_runs} runs)...")
        for i in range(num_runs):
            if self.device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            if self.device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Record memory usage
            if self.device == "cuda":
                memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                memory_usage.append(memory_gb)

            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{num_runs}")

        # Calculate metrics
        latencies = np.array(latencies)
        output_length = outputs.shape[1] - input_length

        result = BenchmarkResult(
            model_name=str(self.model_path),
            model_type="pytorch",
            mean_latency_ms=float(np.mean(latencies)),
            median_latency_ms=float(np.median(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            tokens_per_second=float(output_length * batch_size * 1000 / np.mean(latencies)),
            samples_per_second=float(batch_size * 1000 / np.mean(latencies)),
            peak_memory_gb=float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if self.device == "cuda" else 0.0,
            avg_memory_gb=float(np.mean(memory_usage)) if memory_usage else 0.0,
            num_runs=num_runs,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
        )

        print("✅ Benchmark complete!")

        return result


def quick_benchmark(
    model_path: str,
    device: str = "cuda",
    num_runs: int = 50,
) -> BenchmarkResult:
    """
    Quick benchmark with default settings

    Args:
        model_path: Path to model
        device: Device to run on
        num_runs: Number of runs

    Returns:
        BenchmarkResult: Results
    """
    benchmarker = ModelBenchmark(
        model_path=model_path,
        device=device,
        warmup_runs=5,
    )

    # Default prompts
    prompts = [
        "Write a Python function to calculate fibonacci numbers:",
        "Explain machine learning in simple terms:",
        "What is the capital of France?",
    ]

    result = benchmarker.benchmark(
        prompts=prompts,
        max_new_tokens=50,
        num_runs=num_runs,
        batch_size=1,
    )

    result.print_summary()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark model performance")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of runs")

    args = parser.parse_args()

    quick_benchmark(
        model_path=args.model_path,
        device=args.device,
        num_runs=args.num_runs,
    )
