#!/usr/bin/env python3
"""
GPU Test Script für NVIDIA L40S
Testet grundlegende CUDA Funktionalität und zeigt GPU Information
"""

import subprocess
import sys

def check_nvidia_smi():
    """Prüft ob nvidia-smi verfügbar ist"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("=== NVIDIA SMI Output ===")
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("❌ nvidia-smi nicht gefunden. NVIDIA Treiber installiert?")
        return False

def check_cuda_available():
    """Prüft CUDA Verfügbarkeit"""
    try:
        import torch
        print("\n=== PyTorch CUDA Info ===")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"\n--- GPU {i} ---")
                print(f"Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"Multi Processor Count: {props.multi_processor_count}")
                print(f"CUDA Capability: {props.major}.{props.minor}")

                # Memory usage
                print(f"\nCurrent Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                print(f"Current Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
                print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")

        return torch.cuda.is_available()
    except ImportError:
        print("\n⚠️  PyTorch nicht installiert")
        print("Installation: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

def simple_cuda_test():
    """Führt einen einfachen CUDA Test durch"""
    try:
        import torch

        if not torch.cuda.is_available():
            print("\n❌ CUDA nicht verfügbar für Tests")
            return False

        print("\n=== Simple CUDA Test ===")

        # Erstelle Tensoren auf GPU
        device = torch.device("cuda")
        size = 10000

        print(f"Erstelle {size}x{size} Matrizen...")
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Matrix Multiplikation
        print("Führe Matrix Multiplikation durch...")
        import time
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()

        print(f"✅ Matrix Multiplikation erfolgreich!")
        print(f"Zeit: {(end - start)*1000:.2f} ms")
        print(f"TFLOPS: {(2 * size**3) / (end - start) / 1e12:.2f}")

        return True
    except Exception as e:
        print(f"\n❌ Fehler beim CUDA Test: {e}")
        return False

def benchmark_inference():
    """Benchmark für Inferenz Performance"""
    try:
        import torch
        import time

        if not torch.cuda.is_available():
            return False

        print("\n=== Inference Benchmark ===")
        device = torch.device("cuda")

        # Simuliere ein kleines CNN
        batch_sizes = [1, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            # Simuliere Bild Input (3 Kanäle, 224x224)
            x = torch.randn(batch_size, 3, 224, 224, device=device)

            # Einfaches Conv Layer
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)

            # Warmup
            for _ in range(10):
                _ = conv(x)
            torch.cuda.synchronize()

            # Benchmark
            iterations = 100
            start = time.time()
            for _ in range(iterations):
                _ = conv(x)
            torch.cuda.synchronize()
            end = time.time()

            avg_time = (end - start) / iterations * 1000
            throughput = batch_size / (avg_time / 1000)

            print(f"Batch Size {batch_size:2d}: {avg_time:.2f} ms/batch, {throughput:.1f} samples/sec")

        return True
    except Exception as e:
        print(f"\n❌ Fehler beim Benchmark: {e}")
        return False

def main():
    print("🚀 NVIDIA L40S GPU Test\n")

    # Test nvidia-smi
    if not check_nvidia_smi():
        sys.exit(1)

    # Test CUDA
    if not check_cuda_available():
        print("\n⚠️  Für vollständige Tests PyTorch mit CUDA installieren")
        sys.exit(0)

    # Einfacher CUDA Test
    simple_cuda_test()

    # Inference Benchmark
    benchmark_inference()

    print("\n✅ Alle Tests abgeschlossen!")

if __name__ == "__main__":
    main()
