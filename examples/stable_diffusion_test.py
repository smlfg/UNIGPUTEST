#!/usr/bin/env python3
"""
Stable Diffusion Test für NVIDIA L40S
Testet Image Generation Performance
"""

import torch
import time
from typing import Optional

def check_requirements():
    """Prüft ob diffusers installiert ist"""
    try:
        import diffusers
        print(f"✅ diffusers {diffusers.__version__}")
        return True
    except ImportError:
        print("❌ diffusers nicht installiert")
        print("\nInstallation:")
        print("pip install diffusers transformers accelerate safetensors")
        return False

def test_stable_diffusion_15():
    """Testet Stable Diffusion 1.5"""
    print("\n=== Test: Stable Diffusion 1.5 ===")

    try:
        from diffusers import StableDiffusionPipeline

        model_id = "runwayml/stable-diffusion-v1-5"
        print(f"Lade Modell: {model_id}...")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipe = pipe.to("cuda")

        # Memory
        memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {memory:.2f} GB")

        # Warmup
        print("Warmup...")
        _ = pipe("test", num_inference_steps=1, output_type="np")

        # Benchmark
        prompt = "A beautiful landscape with mountains and a lake, digital art"
        num_images = 5

        print(f"\nGeneriere {num_images} Bilder...")
        print(f"Prompt: '{prompt}'")

        times = []
        for i in range(num_images):
            start = time.time()
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            end = time.time()

            elapsed = end - start
            times.append(elapsed)
            print(f"  Bild {i+1}: {elapsed:.2f}s")

        avg_time = sum(times) / len(times)
        print(f"\nDurchschnitt: {avg_time:.2f}s pro Bild")
        print(f"Throughput: {1/avg_time:.2f} Bilder/Minute")

        # Cleanup
        del pipe
        torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_stable_diffusion_xl():
    """Testet Stable Diffusion XL"""
    print("\n=== Test: Stable Diffusion XL ===")

    try:
        from diffusers import StableDiffusionXLPipeline

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"Lade Modell: {model_id}...")
        print("(Dies kann beim ersten Mal länger dauern)")

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipe = pipe.to("cuda")

        # Memory
        memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {memory:.2f} GB")

        # Warmup
        print("Warmup...")
        _ = pipe("test", num_inference_steps=1, output_type="np")

        # Benchmark
        prompt = "A futuristic city with flying cars, cyberpunk style, highly detailed"
        num_images = 3

        print(f"\nGeneriere {num_images} Bilder (1024x1024)...")
        print(f"Prompt: '{prompt}'")

        times = []
        for i in range(num_images):
            start = time.time()
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=1024,
                width=1024
            ).images[0]
            end = time.time()

            elapsed = end - start
            times.append(elapsed)
            print(f"  Bild {i+1}: {elapsed:.2f}s")

        avg_time = sum(times) / len(times)
        print(f"\nDurchschnitt: {avg_time:.2f}s pro Bild")
        print(f"Throughput: {60/avg_time:.2f} Bilder/Stunde")

        # Cleanup
        del pipe
        torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_batch_sizes():
    """Benchmark verschiedener Batch Sizes"""
    print("\n=== Batch Size Benchmark ===")

    try:
        from diffusers import StableDiffusionPipeline

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipe = pipe.to("cuda")

        # Warmup
        _ = pipe("test", num_inference_steps=1, output_type="np")

        batch_sizes = [1, 2, 4]
        prompt = "A cat sitting on a table"

        print(f"\nPrompt: '{prompt}'")
        print(f"Steps: 20\n")

        for batch_size in batch_sizes:
            try:
                prompts = [prompt] * batch_size

                start = time.time()
                images = pipe(
                    prompts,
                    num_inference_steps=20,
                    height=512,
                    width=512
                ).images
                end = time.time()

                elapsed = end - start
                per_image = elapsed / batch_size
                throughput = batch_size / elapsed

                print(f"Batch Size {batch_size}:")
                print(f"  Total Zeit: {elapsed:.2f}s")
                print(f"  Pro Bild: {per_image:.2f}s")
                print(f"  Throughput: {throughput:.2f} Bilder/s")

                memory = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  Peak Memory: {memory:.2f} GB\n")

                torch.cuda.reset_peak_memory_stats()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Batch Size {batch_size}: ❌ Out of Memory\n")
                    torch.cuda.empty_cache()
                else:
                    raise

        del pipe
        torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def show_recommendations():
    """Zeigt Empfehlungen für Image Generation"""
    print("\n=== Empfehlungen für L40S ===\n")

    print("📊 Performance Expectations:")
    print("  SD 1.5 (512x512):     1-2s pro Bild")
    print("  SDXL (1024x1024):     3-5s pro Bild")
    print("  Batch Size 4:         ~30% schneller pro Bild")
    print()

    print("🎯 Optimierungen:")
    print("  ✓ torch.float16 (Standard)")
    print("  ✓ xformers / flash-attention")
    print("  ✓ torch.compile() für ~20% speedup")
    print("  ✓ TensorRT für Produktion")
    print()

    print("💡 Use Cases:")
    print("  • API Server: 3-5 parallele Requests")
    print("  • Batch Processing: 100+ Bilder/Stunde (SDXL)")
    print("  • Training: LoRA Training gut möglich")
    print("  • Multi-Model: 2-3 Modelle gleichzeitig")
    print()

    print("🔧 Tools:")
    print("  • ComfyUI - Visual Workflow Editor")
    print("  • Automatic1111 - WebUI für SD")
    print("  • InvokeAI - Production-ready Platform")
    print()

def main():
    print("🎨 Stable Diffusion Test für NVIDIA L40S\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA nicht verfügbar!")
        return

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Check requirements
    if not check_requirements():
        show_recommendations()
        return

    # Run tests
    print("\n" + "="*60)
    print("Wähle Test:")
    print("1 - SD 1.5 (schnell)")
    print("2 - SDXL (langsam, hohe Qualität)")
    print("3 - Batch Size Benchmark")
    print("4 - Alle Tests")
    print("="*60)

    choice = input("\nAuswahl (1-4) [4]: ").strip() or "4"

    if choice in ["1", "4"]:
        test_stable_diffusion_15()

    if choice in ["2", "4"]:
        test_stable_diffusion_xl()

    if choice in ["3", "4"]:
        benchmark_batch_sizes()

    show_recommendations()

    print("\n✅ Tests abgeschlossen!")

if __name__ == "__main__":
    main()
