#!/usr/bin/env python3
"""
Model Loading and Configuration Utilities
Handles model initialization, quantization, and LoRA setup
"""

from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_tokenizer(
    model_name: str,
    padding_side: str = "right",
    add_eos_token: bool = True,
) -> PreTrainedTokenizer:
    """
    Load tokenizer with custom settings

    Args:
        model_name: HuggingFace model name
        padding_side: Padding side ('left' or 'right')
        add_eos_token: Add EOS token to sequences

    Returns:
        PreTrainedTokenizer: Loaded tokenizer
    """
    print(f"📚 Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure padding
    tokenizer.padding_side = padding_side

    # Add EOS token to sequences
    if add_eos_token:
        tokenizer.add_eos_token = True

    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")

    return tokenizer


def load_base_model(
    model_name: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    max_memory: Optional[Dict[int, str]] = None,
) -> PreTrainedModel:
    """
    Load base model with optional quantization

    Args:
        model_name: HuggingFace model name
        quantization_config: BitsAndBytes quantization config
        device_map: Device mapping strategy
        torch_dtype: Default dtype for model weights
        max_memory: Maximum memory per device

    Returns:
        PreTrainedModel: Loaded model
    """
    print(f"🤖 Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        max_memory=max_memory,
    )

    print(f"✅ Model loaded successfully")

    # Print model info
    if hasattr(model, "num_parameters"):
        total_params = model.num_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

    return model


def create_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes quantization configuration

    Args:
        load_in_4bit: Use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype for quantized model
        bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4')
        bnb_4bit_use_double_quant: Use double quantization

    Returns:
        BitsAndBytesConfig: Quantization configuration
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    print(f"⚙️  Quantization config created:")
    print(f"   4-bit: {load_in_4bit}")
    print(f"   Compute dtype: {bnb_4bit_compute_dtype}")
    print(f"   Quant type: {bnb_4bit_quant_type}")

    return config


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Create LoRA configuration

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        target_modules: Modules to apply LoRA to
        lora_dropout: Dropout rate for LoRA layers
        bias: Bias training strategy
        task_type: Task type

    Returns:
        LoraConfig: LoRA configuration
    """
    if target_modules is None:
        # Default for Llama models
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

    print(f"🔧 LoRA config created:")
    print(f"   Rank: {r}")
    print(f"   Alpha: {lora_alpha}")
    print(f"   Target modules: {len(target_modules)}")

    return config


def setup_peft_model(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """
    Setup PEFT (LoRA) model for training

    Args:
        model: Base model
        lora_config: LoRA configuration
        gradient_checkpointing: Enable gradient checkpointing

    Returns:
        PreTrainedModel: PEFT model ready for training
    """
    print("🔄 Setting up PEFT model...")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"✅ PEFT model ready!")
    print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"   All params: {all_params:,}")

    return model


def print_model_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"💾 GPU Memory:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        print(f"   Total: {total:.2f} GB")
        print(f"   Free: {total - reserved:.2f} GB")


if __name__ == "__main__":
    print("Model utilities - use in training scripts")
