#!/usr/bin/env python3
"""
GPU Benchmark Suite for NVIDIA L40S
Tests memory bandwidth, compute performance, and mixed precision
"""

import time
import argparse
from typing import Dict, List, Tuple

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def benchmark_pytorch():
    """Benchmark PyTorch operations"""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Skipping PyTorch benchmarks.")
        return

    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run PyTorch benchmarks.")
        return

    print_header("PyTorch Benchmarks")

    device = torch.device('cuda:0')
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")

    # Warm-up
    print("Warming up GPU...")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    for _ in range(10):
        _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    del x, y
    torch.cuda.empty_cache()

    results = {}

    # 1. Matrix Multiplication Benchmarks
    print("\n1. Matrix Multiplication Performance")
    print("-" * 70)

    sizes = [1024, 2048, 4096, 8192]
    dtypes = {
        'FP32': torch.float32,
        'FP16': torch.float16,
        'BF16': torch.bfloat16
    }

    for dtype_name, dtype in dtypes.items():
        print(f"\n{dtype_name} Precision:")
        for size in sizes:
            try:
                a = torch.randn(size, size, device=device, dtype=dtype)
                b = torch.randn(size, size, device=device, dtype=dtype)

                # Benchmark
                torch.cuda.synchronize()
                start = time.perf_counter()

                iterations = 10
                for _ in range(iterations):
                    c = torch.matmul(a, b)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                # Calculate TFLOPS
                flops = 2 * size**3 * iterations  # 2*N^3 for matrix multiply
                tflops = (flops / elapsed) / 1e12

                print(f"  {size}x{size}: {elapsed/iterations*1000:.2f} ms/iter, {tflops:.2f} TFLOPS")

                results[f'matmul_{dtype_name}_{size}'] = {
                    'time_ms': elapsed/iterations*1000,
                    'tflops': tflops
                }

                del a, b, c
                torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"  {size}x{size}: Out of memory")

    # 2. Memory Bandwidth Test
    print("\n\n2. Memory Bandwidth Test")
    print("-" * 70)

    sizes_mb = [100, 500, 1000, 5000, 10000]  # MB
    for size_mb in sizes_mb:
        try:
            elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32

            # Allocation and copy
            torch.cuda.synchronize()
            start = time.perf_counter()

            x = torch.randn(elements, device=device, dtype=torch.float32)
            y = x.clone()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bandwidth_gb_s = (size_mb / 1024) / elapsed
            print(f"  {size_mb:5d} MB: {elapsed*1000:.2f} ms, {bandwidth_gb_s:.2f} GB/s")

            results[f'bandwidth_{size_mb}mb'] = {
                'time_ms': elapsed*1000,
                'bandwidth_gb_s': bandwidth_gb_s
            }

            del x, y
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"  {size_mb:5d} MB: Out of memory")

    # 3. Convolution Benchmark
    print("\n\n3. Convolution Performance")
    print("-" * 70)

    try:
        import torch.nn as nn

        batch_sizes = [1, 8, 16, 32]

        for batch_size in batch_sizes:
            # Typical ResNet-like layer
            conv = nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda()
            x = torch.randn(batch_size, 64, 224, 224, device=device)

            torch.cuda.synchronize()
            start = time.perf_counter()

            iterations = 50
            for _ in range(iterations):
                y = conv(x)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            images_per_sec = (batch_size * iterations) / elapsed

            print(f"  Batch {batch_size:2d}: {elapsed/iterations*1000:.2f} ms/iter, {images_per_sec:.2f} img/s")

            results[f'conv_batch_{batch_size}'] = {
                'time_ms': elapsed/iterations*1000,
                'images_per_sec': images_per_sec
            }

            del conv, x, y
            torch.cuda.empty_cache()

    except ImportError:
        print("  torch.nn not available")

    # 4. Tensor Operations
    print("\n\n4. Element-wise Operations")
    print("-" * 70)

    size = 10000000  # 10M elements
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)

    ops = {
        'Add': lambda: x + y,
        'Multiply': lambda: x * y,
        'Exp': lambda: torch.exp(x),
        'Sin': lambda: torch.sin(x),
        'Sqrt': lambda: torch.sqrt(torch.abs(x))
    }

    for op_name, op_func in ops.items():
        torch.cuda.synchronize()
        start = time.perf_counter()

        iterations = 100
        for _ in range(iterations):
            _ = op_func()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"  {op_name:10s}: {elapsed/iterations*1000:.3f} ms/iter")

        results[f'elemwise_{op_name.lower()}'] = elapsed/iterations*1000

    del x, y
    torch.cuda.empty_cache()

    # Memory Summary
    print("\n\n5. Memory Information")
    print("-" * 70)
    print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"  Allocated:    {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    print(f"  Cached:       {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")

    return results

def benchmark_tensorflow():
    """Benchmark TensorFlow operations"""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. Skipping TensorFlow benchmarks.")
        return

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU available for TensorFlow. Skipping benchmarks.")
        return

    print_header("TensorFlow Benchmarks")

    # Matrix multiplication
    print("\n1. Matrix Multiplication Performance")
    print("-" * 70)

    sizes = [1024, 2048, 4096, 8192]

    for size in sizes:
        try:
            with tf.device('/GPU:0'):
                a = tf.random.normal([size, size])
                b = tf.random.normal([size, size])

                # Warm-up
                _ = tf.matmul(a, b)

                # Benchmark
                start = time.perf_counter()
                iterations = 10

                for _ in range(iterations):
                    c = tf.matmul(a, b)

                elapsed = time.perf_counter() - start

                flops = 2 * size**3 * iterations
                tflops = (flops / elapsed) / 1e12

                print(f"  {size}x{size}: {elapsed/iterations*1000:.2f} ms/iter, {tflops:.2f} TFLOPS")

        except Exception as e:
            print(f"  {size}x{size}: Error - {e}")

    print("\n2. Convolution Performance")
    print("-" * 70)

    batch_sizes = [1, 8, 16, 32]

    for batch_size in batch_sizes:
        try:
            with tf.device('/GPU:0'):
                x = tf.random.normal([batch_size, 224, 224, 64])
                kernel = tf.random.normal([3, 3, 64, 64])

                # Warm-up
                _ = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')

                # Benchmark
                start = time.perf_counter()
                iterations = 50

                for _ in range(iterations):
                    y = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')

                elapsed = time.perf_counter() - start
                images_per_sec = (batch_size * iterations) / elapsed

                print(f"  Batch {batch_size:2d}: {elapsed/iterations*1000:.2f} ms/iter, {images_per_sec:.2f} img/s")

        except Exception as e:
            print(f"  Batch {batch_size}: Error - {e}")

def main():
    """Main benchmark function"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         NVIDIA L40S GPU Benchmark Suite                         ║
    ║         Performance Testing for ML/AI Workloads                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    parser = argparse.ArgumentParser(description='GPU Benchmark Suite')
    parser.add_argument('--framework', choices=['pytorch', 'tensorflow', 'all'],
                       default='all', help='Framework to benchmark')
    args = parser.parse_args()

    if args.framework in ['pytorch', 'all']:
        benchmark_pytorch()

    if args.framework in ['tensorflow', 'all']:
        benchmark_tensorflow()

    print_header("Benchmark Complete")
    print("All tests finished successfully!\n")

if __name__ == "__main__":
    main()
