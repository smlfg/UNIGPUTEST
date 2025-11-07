#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script
Efficient LLM fine-tuning with LoRA and 4-bit quantization
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainingArguments, Trainer
from peft import PeftModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.gpu_check import check_gpu_availability, print_gpu_info
from src.utils.model_utils import (
    load_tokenizer,
    load_base_model,
    create_quantization_config,
    create_lora_config,
    setup_peft_model,
    print_model_memory,
)
from src.training.dataset import DatasetLoader


def main(config_path: Optional[str] = None):
    """
    Main training function

    Args:
        config_path: Path to training configuration YAML
    """
    print("=" * 60)
    print("🚀 LLM FINE-TUNING PIPELINE")
    print("=" * 60)
    print()

    # 1. Check GPU availability
    gpu_info = check_gpu_availability()
    print_gpu_info(gpu_info)

    if not gpu_info["cuda_available"]:
        print("❌ CUDA not available. Exiting.")
        sys.exit(1)

    print()

    # 2. Load configuration
    print("📋 Loading configuration...")
    config_loader = ConfigLoader(config_path)
    config = config_loader.load()
    config_loader.validate()
    print("✅ Configuration loaded successfully!")
    print()

    # 3. Create output directory
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Load tokenizer
    model_name = config["model"]["name"]
    tokenizer = load_tokenizer(model_name)
    print()

    # 5. Load dataset
    print("📦 Loading dataset...")
    dataset_config = config["dataset"]
    dataset_loader = DatasetLoader(
        dataset_name=dataset_config["name"],
        tokenizer=tokenizer,
        max_seq_length=dataset_config["max_seq_length"],
        train_split=dataset_config["split"]["train"],
        eval_split=dataset_config["split"]["eval"],
    )

    datasets = dataset_loader.load()

    # Preprocess dataset
    prompt_template = config.get("prompt", {}).get("template")
    input_template = config.get("prompt", {}).get("input_template")

    datasets = dataset_loader.preprocess(
        prompt_template=prompt_template,
        input_template=input_template
    )
    print()

    # 6. Create quantization config
    quant_config = create_quantization_config(
        load_in_4bit=config["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=config["model"]["bnb_4bit_compute_dtype"],
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["model"]["bnb_4bit_use_double_quant"],
    )
    print()

    # 7. Load base model
    max_memory = config.get("hardware", {}).get("max_memory")
    torch_dtype = getattr(torch, config.get("hardware", {}).get("torch_dtype", "bfloat16"))

    model = load_base_model(
        model_name=model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        max_memory=max_memory,
    )
    print()

    # 8. Setup LoRA
    lora_config = create_lora_config(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
    )
    print()

    # 9. Setup PEFT model
    model = setup_peft_model(
        model=model,
        lora_config=lora_config,
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
    )
    print()

    # 10. Print memory usage
    print_model_memory()
    print()

    # 11. Setup training arguments
    training_config = config["training"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        optim=training_config["optim"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        warmup_ratio=training_config["warmup_ratio"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config.get("eval_steps", 100),
        save_total_limit=training_config["save_total_limit"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        group_by_length=training_config["group_by_length"],
        report_to=training_config.get("report_to", "tensorboard"),
        evaluation_strategy=config.get("evaluation", {}).get("evaluation_strategy", "steps"),
        do_eval=config.get("evaluation", {}).get("do_eval", True),
        seed=config.get("seed", 42),
        logging_dir=str(output_dir / "logs"),
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # 12. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokenizer,
    )

    # 13. Start training
    print("🔥 Starting training...")
    print("=" * 60)
    print()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

    print()
    print("=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print()

    # 14. Save final model
    final_model_path = output_dir / "final_model"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"💾 Model saved to: {final_model_path}")
    print()

    # 15. Print final memory usage
    print_model_memory()
    print()

    print("🎉 All done! You can now:")
    print(f"   1. Check logs: {output_dir / 'logs'}")
    print(f"   2. Load model: {final_model_path}")
    print(f"   3. Export to ONNX for deployment")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training configuration YAML"
    )

    args = parser.parse_args()

    main(config_path=args.config)
