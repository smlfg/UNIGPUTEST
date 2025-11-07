#!/usr/bin/env python3
"""
Test Fine-Tuned Model
Evaluate the fine-tuned model using test prompts

Usage:
    python src/evaluation/test_model.py --model checkpoints/final_model
    python src/evaluation/test_model.py --model checkpoints/final_model --category basic
    python src/evaluation/test_model.py --model checkpoints/final_model --limit 5
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datetime import datetime


def load_model(base_model_name: str, adapter_path: str, device: str = "cuda"):
    """
    Load fine-tuned model with LoRA adapter

    Args:
        base_model_name: Base model name (e.g., "meta-llama/Llama-3.2-1B")
        adapter_path: Path to LoRA adapter
        device: Device to use (cuda/cpu)

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def load_test_prompts(prompts_file: str = "data/test_prompts.json") -> Dict:
    """Load test prompts from JSON file"""
    with open(prompts_file, 'r') as f:
        return json.load(f)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generate response from model

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        str: Generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response


def format_prompt(task: str) -> str:
    """
    Format prompt in Alpaca style

    Args:
        task: The task description

    Returns:
        str: Formatted prompt
    """
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{task}

### Response:
"""


def test_model(
    model,
    tokenizer,
    test_prompts: Dict,
    category: str = "all",
    limit: int = None,
    output_file: str = None
):
    """
    Test model with prompts

    Args:
        model: The model
        tokenizer: The tokenizer
        test_prompts: Dictionary of test prompts
        category: Category to test ('all', 'basic', 'intermediate', 'advanced', etc.)
        limit: Maximum number of prompts to test per category
        output_file: File to save results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "limit": limit,
        "tests": []
    }

    # Get prompts to test
    prompts_to_test = []

    if category == "all":
        for cat_name, prompts in test_prompts["test_prompts"].items():
            prompts_to_test.extend([(cat_name, p) for p in prompts])
    else:
        if category in test_prompts["test_prompts"]:
            prompts_to_test = [(category, p) for p in test_prompts["test_prompts"][category]]
        else:
            print(f"❌ Category '{category}' not found!")
            return

    # Apply limit
    if limit:
        prompts_to_test = prompts_to_test[:limit]

    print(f"\n{'='*80}")
    print(f"🧪 TESTING MODEL")
    print(f"{'='*80}")
    print(f"Category: {category}")
    print(f"Total prompts: {len(prompts_to_test)}")
    print(f"{'='*80}\n")

    # Test each prompt
    for i, (cat_name, prompt_data) in enumerate(prompts_to_test, 1):
        prompt_text = prompt_data["prompt"]
        prompt_id = prompt_data["id"]

        print(f"\n{'─'*80}")
        print(f"Test {i}/{len(prompts_to_test)}")
        print(f"ID: {prompt_id}")
        print(f"Category: {cat_name} | Difficulty: {prompt_data['difficulty']}")
        print(f"{'─'*80}")
        print(f"📝 Prompt: {prompt_text}")
        print(f"{'─'*80}")

        # Format and generate
        formatted_prompt = format_prompt(prompt_text)
        response = generate_response(model, tokenizer, formatted_prompt)

        print(f"🤖 Response:\n{response}")
        print(f"{'─'*80}\n")

        # Save result
        results["tests"].append({
            "id": prompt_id,
            "category": cat_name,
            "difficulty": prompt_data["difficulty"],
            "prompt": prompt_text,
            "response": response
        })

    # Save results to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests run: {len(results['tests'])}")
    print(f"Category: {category}")
    print(f"Results saved: {output_file or 'Not saved'}")
    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Fine-Tuned Model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to LoRA adapter (e.g., checkpoints/final_model)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name (default: meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/test_prompts.json",
        help="Path to test prompts JSON file"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "basic", "intermediate", "advanced", "specialized", "real_world"],
        help="Category of prompts to test"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of prompts to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: results/test_results_TIMESTAMP.json)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.base_model, args.model)

    # Load test prompts
    test_prompts = load_test_prompts(args.prompts)

    # Set default output file
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/test_results_{args.category}_{timestamp}.json"

    # Test model
    test_model(
        model,
        tokenizer,
        test_prompts,
        category=args.category,
        limit=args.limit,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
