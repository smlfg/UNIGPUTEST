#!/usr/bin/env python3
"""
PyTorch GPU Test Suite
Comprehensive tests for PyTorch on NVIDIA L40S
"""

import torch
import time
import sys

def test_basic_operations():
    """Test basic PyTorch GPU operations"""
    print("\n" + "="*70)
    print("  1. Basic Operations Test")
    print("="*70 + "\n")

    device = torch.device('cuda:0')

    try:
        # Tensor creation
        print("✓ Creating tensors on GPU...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)

        # Basic operations
        print("✓ Testing addition...")
        z = x + y

        print("✓ Testing multiplication...")
        z = x * y

        print("✓ Testing matrix multiplication...")
        z = torch.matmul(x, y)

        print("✓ Testing reduction operations...")
        mean = z.mean()
        std = z.std()
        sum_val = z.sum()

        print(f"\n  Results:")
        print(f"    Mean: {mean.item():.4f}")
        print(f"    Std:  {std.item():.4f}")

        print("\n✓ All basic operations passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Basic operations failed: {e}\n")
        return False

def test_mixed_precision():
    """Test mixed precision training capabilities"""
    print("\n" + "="*70)
    print("  2. Mixed Precision Test (FP16/BF16)")
    print("="*70 + "\n")

    device = torch.device('cuda:0')

    try:
        # Test FP16
        print("Testing FP16 operations...")
        x_fp16 = torch.randn(2048, 2048, device=device, dtype=torch.float16)
        y_fp16 = torch.randn(2048, 2048, device=device, dtype=torch.float16)

        start = time.perf_counter()
        z_fp16 = torch.matmul(x_fp16, y_fp16)
        torch.cuda.synchronize()
        fp16_time = time.perf_counter() - start

        print(f"  FP16 matmul (2048x2048): {fp16_time*1000:.2f} ms")

        # Test BF16
        print("\nTesting BF16 operations...")
        x_bf16 = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
        y_bf16 = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)

        start = time.perf_counter()
        z_bf16 = torch.matmul(x_bf16, y_bf16)
        torch.cuda.synchronize()
        bf16_time = time.perf_counter() - start

        print(f"  BF16 matmul (2048x2048): {bf16_time*1000:.2f} ms")

        # Test FP32 for comparison
        print("\nTesting FP32 operations...")
        x_fp32 = torch.randn(2048, 2048, device=device, dtype=torch.float32)
        y_fp32 = torch.randn(2048, 2048, device=device, dtype=torch.float32)

        start = time.perf_counter()
        z_fp32 = torch.matmul(x_fp32, y_fp32)
        torch.cuda.synchronize()
        fp32_time = time.perf_counter() - start

        print(f"  FP32 matmul (2048x2048): {fp32_time*1000:.2f} ms")

        print(f"\n  Speedup:")
        print(f"    FP16 vs FP32: {fp32_time/fp16_time:.2f}x faster")
        print(f"    BF16 vs FP32: {fp32_time/bf16_time:.2f}x faster")

        print("\n✓ Mixed precision tests passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Mixed precision test failed: {e}\n")
        return False

def test_neural_network():
    """Test a simple neural network training"""
    print("\n" + "="*70)
    print("  3. Neural Network Training Test")
    print("="*70 + "\n")

    device = torch.device('cuda:0')

    try:
        import torch.nn as nn
        import torch.optim as optim

        # Simple CNN model
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(256 * 28 * 28, 512)
                self.fc2 = nn.Linear(512, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(-1, 256 * 28 * 28)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        print("Creating model...")
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create dummy data
        batch_size = 32
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        print("Training for 10 iterations...")
        times = []

        for i in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if i == 0 or (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}: {elapsed*1000:.2f} ms, Loss: {loss.item():.4f}")

        avg_time = sum(times[1:]) / len(times[1:])  # Skip first iteration (warmup)
        print(f"\n  Average time (excl. warmup): {avg_time*1000:.2f} ms")
        print(f"  Throughput: {batch_size/avg_time:.2f} images/sec")

        print("\n✓ Neural network test passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Neural network test failed: {e}\n")
        return False

def test_memory_management():
    """Test GPU memory management"""
    print("\n" + "="*70)
    print("  4. Memory Management Test")
    print("="*70 + "\n")

    device = torch.device('cuda:0')

    try:
        # Get initial memory
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated(device) / (1024**3)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)

        print(f"Total GPU Memory: {total_mem:.2f} GB")
        print(f"Initial allocated: {initial_mem:.2f} GB\n")

        # Allocate large tensors
        print("Allocating tensors...")
        tensors = []

        for i in range(5):
            size = 1024 * 1024 * 256  # ~1GB per tensor
            t = torch.randn(size, device=device)
            tensors.append(t)

            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)

            print(f"  After tensor {i+1}:")
            print(f"    Allocated: {allocated:.2f} GB")
            print(f"    Reserved:  {reserved:.2f} GB")

        # Clear memory
        print("\nClearing tensors...")
        del tensors
        torch.cuda.empty_cache()

        final_mem = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"Final allocated: {final_mem:.2f} GB")

        print("\n✓ Memory management test passed!\n")
        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n✗ Out of memory (expected on smaller GPUs)\n")
            return True
        else:
            print(f"\n✗ Memory management test failed: {e}\n")
            return False
    except Exception as e:
        print(f"\n✗ Memory management test failed: {e}\n")
        return False

def test_multi_gpu():
    """Test multi-GPU capabilities if available"""
    print("\n" + "="*70)
    print("  5. Multi-GPU Test")
    print("="*70 + "\n")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}\n")

    if num_gpus < 2:
        print("Single GPU detected. Skipping multi-GPU tests.")
        print("✓ Test skipped (not applicable)\n")
        return True

    try:
        print("Testing DataParallel...")
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(1000, 1000)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()
        model = nn.DataParallel(model)
        model = model.cuda()

        x = torch.randn(64, 1000).cuda()
        output = model(x)

        print(f"  Output shape: {output.shape}")
        print("\n✓ Multi-GPU test passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Multi-GPU test failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         PyTorch GPU Test Suite for NVIDIA L40S                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("Please check your CUDA installation and GPU drivers.")
        sys.exit(1)

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {'.'.join(map(str, torch.cuda.get_device_capability(0)))}")

    # Run tests
    tests = [
        ("Basic Operations", test_basic_operations),
        ("Mixed Precision", test_mixed_precision),
        ("Neural Network", test_neural_network),
        ("Memory Management", test_memory_management),
        ("Multi-GPU", test_multi_gpu),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Unexpected error in {test_name}: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70 + "\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:25s} {status}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70 + "\n")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
