#!/usr/bin/env python3
"""
Advanced LLM Benchmark Suite for NVIDIA L40S
Comprehensive testing with multiple models, quantizations, and metrics
"""

import torch
import time
import json
import yaml
import argparse
import subprocess
import psutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import threading
import queue

# Try to import optional dependencies
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    model_name: str
    quantization: str
    batch_sizes: List[int]
    sequence_length: int
    num_iterations: int
    warmup_iterations: int
    output_dir: str
    monitor_gpu: bool = True
    monitor_interval: float = 0.1


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark"""
    model_name: str
    quantization: str
    batch_size: int
    timestamp: str

    # Performance metrics
    tokens_per_second: float
    latency_ms: float
    throughput_tokens: float

    # Memory metrics
    memory_allocated_gb: float
    memory_reserved_gb: float
    memory_peak_gb: float
    memory_fragmentation: float

    # GPU metrics
    gpu_utilization_avg: float
    gpu_utilization_peak: float
    temperature_avg: float
    temperature_peak: float
    power_avg_watts: float
    power_peak_watts: float

    # Cache metrics
    cache_hit_rate: float = 0.0

    # Additional info
    model_parameters: int = 0
    device_name: str = ""
    cuda_version: str = ""


class GPUMonitor:
    """Real-time GPU monitoring using NVIDIA Management Library"""

    def __init__(self, device_id: int = 0, interval: float = 0.1):
        self.device_id = device_id
        self.interval = interval
        self.monitoring = False
        self.data_queue = queue.Queue()
        self.monitor_thread = None

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.available = True
            except Exception as e:
                print(f"⚠️  NVML initialization failed: {e}")
                self.available = False
        else:
            self.available = False

    def start(self):
        """Start monitoring in background thread"""
        if not self.available:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics"""
        if not self.available:
            return {}

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        # Collect all measurements
        measurements = []
        while not self.data_queue.empty():
            measurements.append(self.data_queue.get())

        if not measurements:
            return {}

        # Calculate statistics
        stats = {
            'utilization_avg': sum(m['utilization'] for m in measurements) / len(measurements),
            'utilization_peak': max(m['utilization'] for m in measurements),
            'temperature_avg': sum(m['temperature'] for m in measurements) / len(measurements),
            'temperature_peak': max(m['temperature'] for m in measurements),
            'power_avg': sum(m['power'] for m in measurements) / len(measurements),
            'power_peak': max(m['power'] for m in measurements),
            'memory_used_avg': sum(m['memory_used'] for m in measurements) / len(measurements),
            'memory_used_peak': max(m['memory_used'] for m in measurements),
        }

        return stats

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to Watts
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                self.data_queue.put({
                    'timestamp': time.time(),
                    'utilization': util.gpu,
                    'temperature': temp,
                    'power': power,
                    'memory_used': mem_info.used / 1024**3,  # GB
                })
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                break

            time.sleep(self.interval)

    def __del__(self):
        if self.available and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class LLMBenchmark:
    """Main LLM Benchmark Suite"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.model_configs = self._load_model_configs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            self.device_name = torch.cuda.get_device_name(0)
            self.cuda_version = torch.version.cuda
        else:
            raise RuntimeError("CUDA not available!")

    def _load_model_configs(self) -> Dict:
        """Load model configurations from YAML file"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def get_quantization_config(self, quant_type: str) -> Optional[Dict]:
        """Get quantization configuration"""
        configs = {
            'int8': {
                'load_in_8bit': True,
                'llm_int8_threshold': 6.0,
            },
            'int4': {
                'load_in_4bit': True,
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': 'nf4',
            },
            'fp16': {
                'torch_dtype': torch.float16,
            },
            'bf16': {
                'torch_dtype': torch.bfloat16,
            },
            'fp32': {
                'torch_dtype': torch.float32,
            },
        }
        return configs.get(quant_type)

    def load_model(self, model_name: str, quantization: str = 'fp16'):
        """Load model with specified quantization"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")

        print(f"\n🔄 Loading {model_name} with {quantization} quantization...")

        quant_config = self.get_quantization_config(quantization)
        if quant_config is None:
            raise ValueError(f"Unknown quantization type: {quantization}")

        # Prepare loading arguments
        load_kwargs = {'device_map': 'auto'}

        if 'load_in_8bit' in quant_config or 'load_in_4bit' in quant_config:
            # BitsAndBytes quantization
            load_kwargs['quantization_config'] = BitsAndBytesConfig(**quant_config)
        elif 'torch_dtype' in quant_config:
            load_kwargs['torch_dtype'] = quant_config['torch_dtype']

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **load_kwargs
            )

            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model.eval()

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())

            print(f"✅ Model loaded: {total_params/1e6:.1f}M parameters")

            return model, tokenizer, total_params

        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None, None, 0

    def apply_dynamic_quantization(self, model):
        """Apply PyTorch dynamic quantization"""
        print("🔄 Applying dynamic quantization...")
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("✅ Dynamic quantization applied")
            return quantized_model
        except Exception as e:
            print(f"⚠️  Dynamic quantization failed: {e}")
            return model

    def calculate_memory_fragmentation(self) -> float:
        """Calculate GPU memory fragmentation"""
        if not torch.cuda.is_available():
            return 0.0

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        if reserved == 0:
            return 0.0

        fragmentation = (reserved - allocated) / reserved
        return fragmentation

    def benchmark_inference(
        self,
        model,
        tokenizer,
        config: BenchmarkConfig,
        monitor: Optional[GPUMonitor] = None
    ) -> List[BenchmarkMetrics]:
        """Run inference benchmark with multiple batch sizes"""

        results = []
        prompt = "The future of artificial intelligence and machine learning will transform"

        for batch_size in config.batch_sizes:
            print(f"\n📊 Testing batch size: {batch_size}")

            # Prepare batched input
            prompts = [prompt] * batch_size
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.sequence_length
            ).to(self.device)

            # Warmup
            print("  Warming up...")
            with torch.no_grad():
                for _ in range(config.warmup_iterations):
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Start monitoring
            if monitor:
                monitor.start()

            # Benchmark
            print("  Running benchmark...")
            latencies = []

            for i in range(config.num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                torch.cuda.synchronize()
                end = time.perf_counter()

                latencies.append((end - start) * 1000)  # Convert to ms

            # Stop monitoring
            monitor_stats = monitor.stop() if monitor else {}

            # Calculate metrics
            avg_latency = sum(latencies) / len(latencies)
            tokens_generated = 50 * batch_size  # per iteration
            tokens_per_second = (tokens_generated * config.num_iterations) / (sum(latencies) / 1000)

            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_peak = torch.cuda.max_memory_allocated() / 1024**3
            memory_frag = self.calculate_memory_fragmentation()

            # Create metrics object
            metrics = BenchmarkMetrics(
                model_name=config.model_name,
                quantization=config.quantization,
                batch_size=batch_size,
                timestamp=datetime.now().isoformat(),
                tokens_per_second=tokens_per_second,
                latency_ms=avg_latency,
                throughput_tokens=tokens_per_second,
                memory_allocated_gb=memory_allocated,
                memory_reserved_gb=memory_reserved,
                memory_peak_gb=memory_peak,
                memory_fragmentation=memory_frag,
                gpu_utilization_avg=monitor_stats.get('utilization_avg', 0.0),
                gpu_utilization_peak=monitor_stats.get('utilization_peak', 0.0),
                temperature_avg=monitor_stats.get('temperature_avg', 0.0),
                temperature_peak=monitor_stats.get('temperature_peak', 0.0),
                power_avg_watts=monitor_stats.get('power_avg', 0.0),
                power_peak_watts=monitor_stats.get('power_peak', 0.0),
                device_name=self.device_name,
                cuda_version=self.cuda_version or "",
            )

            results.append(metrics)

            # Print summary
            print(f"\n  Results:")
            print(f"    Latency:        {avg_latency:.2f} ms")
            print(f"    Throughput:     {tokens_per_second:.2f} tokens/s")
            print(f"    Memory Peak:    {memory_peak:.2f} GB")
            print(f"    Fragmentation:  {memory_frag*100:.1f}%")
            if monitor_stats:
                print(f"    GPU Util:       {monitor_stats.get('utilization_avg', 0):.1f}%")
                print(f"    Temperature:    {monitor_stats.get('temperature_avg', 0):.1f}°C")
                print(f"    Power:          {monitor_stats.get('power_avg', 0):.1f}W")

            # Cleanup
            del outputs
            torch.cuda.empty_cache()

        return results

    def run_benchmark(
        self,
        model_name: str,
        quantization: str = 'fp16',
        batch_sizes: List[int] = [1, 4, 8, 16],
        output_dir: str = 'benchmark_results'
    ) -> List[BenchmarkMetrics]:
        """Run complete benchmark for a model"""

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create configuration
        config = BenchmarkConfig(
            model_name=model_name,
            quantization=quantization,
            batch_sizes=batch_sizes,
            sequence_length=512,
            num_iterations=10,
            warmup_iterations=3,
            output_dir=output_dir,
            monitor_gpu=True
        )

        # Load model
        model, tokenizer, total_params = self.load_model(model_name, quantization)
        if model is None:
            return []

        # Special handling for dynamic quantization
        if quantization == 'dynamic':
            model = self.apply_dynamic_quantization(model)

        # Create GPU monitor
        monitor = GPUMonitor() if config.monitor_gpu else None

        # Run benchmark
        results = self.benchmark_inference(model, tokenizer, config, monitor)

        # Update parameter count
        for r in results:
            r.model_parameters = total_params

        # Save results
        self.save_results(results, output_dir)

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

        return results

    def save_results(self, results: List[BenchmarkMetrics], output_dir: str):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = results[0].model_name.replace('/', '_')

        # Save as JSON
        json_path = os.path.join(
            output_dir,
            f"{model_safe_name}_{results[0].quantization}_{timestamp}.json"
        )
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        print(f"\n💾 Results saved to: {json_path}")

    def run_model_suite(
        self,
        models: List[str],
        quantizations: List[str],
        batch_sizes: List[int] = [1, 4, 8, 16],
        output_dir: str = 'benchmark_results'
    ):
        """Run benchmarks for multiple models and quantizations"""

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         Advanced LLM Benchmark Suite                            ║
║         NVIDIA L40S Performance Testing                         ║
╚══════════════════════════════════════════════════════════════════╝

Device: {self.device_name}
CUDA:   {self.cuda_version}
Models: {len(models)}
Quantizations: {', '.join(quantizations)}
Batch Sizes: {batch_sizes}
        """)

        all_results = []

        for model_name in models:
            for quantization in quantizations:
                print(f"\n{'='*70}")
                print(f"🚀 Benchmarking: {model_name} ({quantization})")
                print(f"{'='*70}")

                try:
                    results = self.run_benchmark(
                        model_name,
                        quantization,
                        batch_sizes,
                        output_dir
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"❌ Error benchmarking {model_name} ({quantization}): {e}")
                    continue

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Advanced LLM Benchmark Suite')
    parser.add_argument('--config', type=str, help='Path to model config YAML')
    parser.add_argument('--model', type=str, help='Single model to benchmark')
    parser.add_argument('--quantization', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'bf16', 'int8', 'int4', 'dynamic'],
                       help='Quantization type')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with single small model')

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = LLMBenchmark(config_path=args.config)

    if args.quick:
        # Quick test with GPT-2
        print("🏃 Running quick test with GPT-2...")
        benchmark.run_benchmark(
            model_name='gpt2',
            quantization='fp16',
            batch_sizes=[1, 4],
            output_dir=args.output_dir
        )
    elif args.model:
        # Single model benchmark
        benchmark.run_benchmark(
            model_name=args.model,
            quantization=args.quantization,
            batch_sizes=args.batch_sizes,
            output_dir=args.output_dir
        )
    else:
        # Full suite - models from config or defaults
        models = [
            'gpt2',
            'tiiuae/falcon-7b',
            'mistralai/Mistral-7B-v0.1',
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'codellama/CodeLlama-7b-hf',
            'codellama/CodeLlama-34b-hf',
        ]

        quantizations = ['fp16', 'int8']

        benchmark.run_model_suite(
            models=models,
            quantizations=quantizations,
            batch_sizes=args.batch_sizes,
            output_dir=args.output_dir
        )

    print("\n✅ Benchmark suite completed!")


if __name__ == "__main__":
    main()
