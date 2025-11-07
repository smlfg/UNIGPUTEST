#!/usr/bin/env python3
"""
Quick Test Script
Verifies installation and GPU availability
"""

import sys


def test_pytorch():
    """Test PyTorch installation"""
    print("🔍 Testing PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

    return True


def test_transformers():
    """Test Transformers installation"""
    print("\n🔍 Testing Transformers...")
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} installed")
        return True
    except ImportError:
        print("❌ Transformers not installed")
        return False


def test_peft():
    """Test PEFT installation"""
    print("\n🔍 Testing PEFT...")
    try:
        import peft
        print(f"✅ PEFT {peft.__version__} installed")
        return True
    except ImportError:
        print("❌ PEFT not installed")
        return False


def test_bitsandbytes():
    """Test BitsAndBytes installation"""
    print("\n🔍 Testing BitsAndBytes...")
    try:
        import bitsandbytes
        print(f"✅ BitsAndBytes {bitsandbytes.__version__} installed")
        return True
    except ImportError:
        print("❌ BitsAndBytes not installed")
        return False


def test_datasets():
    """Test Datasets installation"""
    print("\n🔍 Testing Datasets...")
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__} installed")
        return True
    except ImportError:
        print("❌ Datasets not installed")
        return False


def test_onnx():
    """Test ONNX installation"""
    print("\n🔍 Testing ONNX...")
    try:
        import onnx
        import onnxruntime
        print(f"✅ ONNX {onnx.__version__} installed")
        print(f"✅ ONNX Runtime {onnxruntime.__version__} installed")
        return True
    except ImportError as e:
        print(f"❌ ONNX not fully installed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 LLM Pipeline Installation Test")
    print("=" * 60)
    print()

    results = {
        "PyTorch": test_pytorch(),
        "Transformers": test_transformers(),
        "PEFT": test_peft(),
        "BitsAndBytes": test_bitsandbytes(),
        "Datasets": test_datasets(),
        "ONNX": test_onnx(),
    }

    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component:20s} {status}")

    print("=" * 60)

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. python src/training/train.py")
        print("  2. jupyter notebook notebooks/quick_start_demo.ipynb")
        return 0
    else:
        print("\n⚠️ Some tests failed. Run setup.sh to install missing dependencies:")
        print("  ./setup.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
