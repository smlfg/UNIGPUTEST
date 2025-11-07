#!/usr/bin/env python3
"""
GPU Availability and VRAM Check
Tests CUDA, displays GPU info, and validates setup
"""

import sys
from typing import Dict, Any


def check_gpu_availability() -> Dict[str, Any]:
    """
    Comprehensive GPU availability check

    Returns:
        dict: GPU information and availability status
    """
    info = {
        "pytorch_available": False,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "cuda_version": None,
        "vram_total_gb": 0,
        "vram_free_gb": 0,
    }

    try:
        import torch
        info["pytorch_available"] = True
        info["pytorch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda

            # Get GPU 0 details
            if info["gpu_count"] > 0:
                info["gpu_name"] = torch.cuda.get_device_name(0)

                # VRAM info
                vram_total = torch.cuda.get_device_properties(0).total_memory
                vram_reserved = torch.cuda.memory_reserved(0)
                vram_allocated = torch.cuda.memory_allocated(0)

                info["vram_total_gb"] = vram_total / (1024**3)
                info["vram_reserved_gb"] = vram_reserved / (1024**3)
                info["vram_allocated_gb"] = vram_allocated / (1024**3)
                info["vram_free_gb"] = (vram_total - vram_reserved) / (1024**3)

    except ImportError:
        info["error"] = "PyTorch not installed"
    except Exception as e:
        info["error"] = str(e)

    return info


def print_gpu_info(info: Dict[str, Any]) -> None:
    """Pretty print GPU information"""
    print("=" * 60)
    print("🖥️  GPU AVAILABILITY CHECK")
    print("=" * 60)

    if not info["pytorch_available"]:
        print("❌ PyTorch not installed")
        print("   Run: pip install torch torchvision torchaudio")
        return

    print(f"✅ PyTorch Version: {info.get('pytorch_version', 'N/A')}")

    if not info["cuda_available"]:
        print("❌ CUDA not available")
        print("   - Check NVIDIA drivers")
        print("   - Ensure PyTorch installed with CUDA support")
        return

    print(f"✅ CUDA Available: True")
    print(f"✅ CUDA Version: {info['cuda_version']}")
    print(f"✅ GPU Count: {info['gpu_count']}")
    print()
    print("GPU 0 Details:")
    print(f"  └─ Name: {info['gpu_name']}")
    print(f"  └─ VRAM Total: {info['vram_total_gb']:.2f} GB")
    print(f"  └─ VRAM Free: {info['vram_free_gb']:.2f} GB")
    print(f"  └─ VRAM Reserved: {info.get('vram_reserved_gb', 0):.2f} GB")
    print(f"  └─ VRAM Allocated: {info.get('vram_allocated_gb', 0):.2f} GB")
    print("=" * 60)

    # Recommendations
    if info['vram_total_gb'] >= 40:
        print("✅ Hardware Ready: Sufficient VRAM for Llama 3.2 3B/8B fine-tuning")
    elif info['vram_total_gb'] >= 24:
        print("⚠️  Medium VRAM: Use QLoRA for larger models")
    else:
        print("⚠️  Low VRAM: Stick to smaller models or aggressive quantization")
    print("=" * 60)


if __name__ == "__main__":
    info = check_gpu_availability()
    print_gpu_info(info)

    # Exit with error if GPU not available
    if not info["cuda_available"]:
        sys.exit(1)
