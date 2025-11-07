#!/usr/bin/env python3
"""
LLM Inference Test für NVIDIA L40S
Zeigt wie man verschiedene LLMs laden und verwenden kann
"""

import torch
import time
from typing import Optional

def check_requirements():
    """Prüft ob notwendige Pakete installiert sind"""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers'
    }

    missing = []
    for package, name in required.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"❌ Fehlende Pakete: {', '.join(missing)}")
        print("\nInstallation:")
        print("pip install torch transformers accelerate bitsandbytes")
        return False

    return True

def test_small_model():
    """Testet ein kleines Modell (GPT-2)"""
    print("\n=== Test: GPT-2 (Small Model) ===")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "gpt2"
        print(f"Lade Modell: {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

        # Memory Usage
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory Used: {memory_used:.2f} GB")

        # Inference Test
        prompt = "The future of artificial intelligence is"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        print(f"\nPrompt: '{prompt}'")
        print("Generiere Text...")

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        end = time.time()

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
        tokens_per_sec = tokens_generated / (end - start)

        print(f"\nGenerated: {generated_text}")
        print(f"\nPerformance:")
        print(f"  Zeit: {(end-start):.2f}s")
        print(f"  Tokens: {tokens_generated}")
        print(f"  Tokens/sec: {tokens_per_sec:.2f}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def estimate_model_memory(model_name: str, dtype: str = "float16"):
    """Schätzt Memory Bedarf für ein Modell"""

    # Bekannte Modell-Parameter
    model_params = {
        "gpt2": 124,  # Million
        "gpt2-medium": 355,
        "gpt2-large": 774,
        "gpt2-xl": 1558,
        "facebook/opt-1.3b": 1300,
        "facebook/opt-6.7b": 6700,
        "meta-llama/Llama-2-7b-hf": 7000,
        "meta-llama/Llama-2-13b-hf": 13000,
        "meta-llama/Llama-2-70b-hf": 70000,
        "mistralai/Mistral-7B-v0.1": 7000,
    }

    params = model_params.get(model_name, None)
    if params is None:
        return None

    # Bytes pro Parameter
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5
    }

    bytes_pp = bytes_per_param.get(dtype, 2)
    memory_gb = (params * 1e6 * bytes_pp) / 1024**3

    # Add overhead (ca. 20% für Activations, etc.)
    memory_gb *= 1.2

    return memory_gb

def show_model_recommendations():
    """Zeigt Empfehlungen für Modelle auf L40S"""
    print("\n=== Modell Empfehlungen für L40S (48GB) ===\n")

    models = [
        ("gpt2", "float16"),
        ("meta-llama/Llama-2-7b-hf", "float16"),
        ("meta-llama/Llama-2-7b-hf", "int8"),
        ("meta-llama/Llama-2-13b-hf", "float16"),
        ("meta-llama/Llama-2-13b-hf", "int8"),
        ("meta-llama/Llama-2-70b-hf", "int8"),
        ("meta-llama/Llama-2-70b-hf", "int4"),
        ("mistralai/Mistral-7B-v0.1", "float16"),
    ]

    print(f"{'Modell':<40} {'Dtype':<10} {'Est. Memory':<12} {'Status'}")
    print("-" * 80)

    for model, dtype in models:
        memory = estimate_model_memory(model, dtype)
        if memory:
            status = "✅ Passt" if memory < 45 else "⚠️  Knapp" if memory < 48 else "❌ Zu groß"
            print(f"{model:<40} {dtype:<10} {memory:>6.1f} GB      {status}")

    print("\n💡 Tipps:")
    print("  - Nutze int8/int4 Quantisierung für größere Modelle")
    print("  - bitsandbytes für automatische Quantisierung")
    print("  - Flash Attention 2 für bessere Memory Effizienz")
    print("  - Gradient Checkpointing für Training")

def test_quantization():
    """Zeigt 8-bit Quantisierung mit bitsandbytes"""
    print("\n=== Test: 8-bit Quantisierung ===")

    try:
        import bitsandbytes
        print(f"✅ bitsandbytes {bitsandbytes.__version__} verfügbar")

        print("\nBeispiel Code für 8-bit Loading:")
        print("""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
        """)

        return True
    except ImportError:
        print("⚠️  bitsandbytes nicht installiert")
        print("Installation: pip install bitsandbytes")
        return False

def show_inference_frameworks():
    """Zeigt verschiedene Inference Frameworks"""
    print("\n=== Inference Frameworks für L40S ===\n")

    frameworks = [
        {
            "name": "vLLM",
            "description": "High-throughput LLM serving",
            "use_case": "Production API serving",
            "install": "pip install vllm",
            "example": "vllm serve meta-llama/Llama-2-7b-hf"
        },
        {
            "name": "Text Generation Inference",
            "description": "HuggingFace LLM server",
            "use_case": "Production serving mit Features",
            "install": "docker pull ghcr.io/huggingface/text-generation-inference",
            "example": "docker run --gpus all -p 8080:80 ..."
        },
        {
            "name": "TensorRT-LLM",
            "description": "NVIDIA optimierte Inferenz",
            "use_case": "Maximale Performance",
            "install": "Siehe NVIDIA docs",
            "example": "Kompilierung erforderlich"
        },
        {
            "name": "llama.cpp (CUDA)",
            "description": "C++ Implementation",
            "use_case": "Einfaches Deployment",
            "install": "git clone + cmake",
            "example": "./main -m model.gguf -ngl 99"
        }
    ]

    for fw in frameworks:
        print(f"📦 {fw['name']}")
        print(f"   {fw['description']}")
        print(f"   Use Case: {fw['use_case']}")
        print(f"   Install: {fw['install']}")
        print(f"   Example: {fw['example']}")
        print()

def main():
    print("🚀 LLM Inference Test für NVIDIA L40S\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA nicht verfügbar!")
        return

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Check requirements
    if not check_requirements():
        show_model_recommendations()
        show_inference_frameworks()
        return

    # Run tests
    test_small_model()
    test_quantization()
    show_model_recommendations()
    show_inference_frameworks()

    print("\n✅ Tests abgeschlossen!")
    print("\n💡 Nächste Schritte:")
    print("   1. Installiere gewünschte Modelle von HuggingFace")
    print("   2. Teste mit verschiedenen Quantisierungen")
    print("   3. Nutze vLLM für Production Serving")

if __name__ == "__main__":
    main()
