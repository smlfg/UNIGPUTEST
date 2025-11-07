#!/usr/bin/env python3
"""
GPU Information and Verification Script
Tests NVIDIA L40S GPU setup and displays detailed information
"""

import subprocess
import sys

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def run_command(cmd, description):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

def check_nvidia_smi():
    """Check if nvidia-smi is available and working"""
    print_section("NVIDIA Driver & GPU Detection")
    output = run_command("nvidia-smi", "NVIDIA SMI")
    print(output)
    return "NVIDIA-SMI" in output

def check_cuda():
    """Check CUDA installation"""
    print_section("CUDA Information")

    # Check nvcc
    nvcc_output = run_command("nvcc --version", "NVCC Version")
    print("NVCC Version:")
    print(nvcc_output)

    # Check CUDA path
    cuda_path = run_command("echo $CUDA_HOME", "CUDA Home")
    print(f"\nCUDA_HOME: {cuda_path.strip() if cuda_path.strip() else 'Not set'}")

    return "release" in nvcc_output.lower()

def check_python_packages():
    """Check if GPU-related Python packages are installed"""
    print_section("Python GPU Packages")

    packages = {
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'cupy': 'CuPy',
        'numba': 'Numba'
    }

    installed = {}
    for pkg, name in packages.items():
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            installed[name] = version
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: Not installed")
            installed[name] = None

    return installed

def test_pytorch_gpu():
    """Test PyTorch GPU access"""
    print_section("PyTorch GPU Test")

    try:
        import torch

        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")

                # Memory info
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  Total Memory: {total_mem:.2f} GB")

                # Test tensor creation
                try:
                    x = torch.randn(1000, 1000, device=f'cuda:{i}')
                    y = torch.randn(1000, 1000, device=f'cuda:{i}')
                    z = torch.matmul(x, y)
                    print(f"  ✓ Tensor operations working")
                    del x, y, z
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  ✗ Tensor operations failed: {e}")

            return True
        else:
            print("CUDA not available in PyTorch")
            return False

    except ImportError:
        print("PyTorch not installed. Install with: pip install torch torchvision torchaudio")
        return False
    except Exception as e:
        print(f"Error testing PyTorch: {e}")
        return False

def test_tensorflow_gpu():
    """Test TensorFlow GPU access"""
    print_section("TensorFlow GPU Test")

    try:
        import tensorflow as tf

        print(f"TensorFlow Version: {tf.__version__}")

        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs Available: {len(gpus)}")

        if gpus:
            for gpu in gpus:
                print(f"  {gpu}")
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Details: {details}")

            # Test operation
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                print("  ✓ Tensor operations working")
                return True
            except Exception as e:
                print(f"  ✗ Tensor operations failed: {e}")
                return False
        else:
            print("No GPUs available in TensorFlow")
            return False

    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"Error testing TensorFlow: {e}")
        return False

def get_gpu_specs():
    """Get detailed GPU specifications"""
    print_section("Detailed GPU Specifications")

    specs = run_command(
        "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,power.draw,power.limit,compute_cap --format=csv",
        "GPU Specs"
    )
    print(specs)

def main():
    """Main function to run all checks"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         NVIDIA L40S GPU Testing Suite                   ║
    ║         GPU Verification & Information Script           ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Run checks
    has_driver = check_nvidia_smi()
    has_cuda = check_cuda()
    installed_packages = check_python_packages()

    if has_driver:
        get_gpu_specs()

    # Test frameworks
    if installed_packages.get('PyTorch'):
        test_pytorch_gpu()

    if installed_packages.get('TensorFlow'):
        test_tensorflow_gpu()

    # Summary
    print_section("Summary")
    print(f"✓ NVIDIA Driver: {'Installed' if has_driver else 'Not Found'}")
    print(f"✓ CUDA Toolkit: {'Installed' if has_cuda else 'Not Found'}")
    print(f"✓ PyTorch: {installed_packages.get('PyTorch', 'Not Installed')}")
    print(f"✓ TensorFlow: {installed_packages.get('TensorFlow', 'Not Installed')}")

    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
