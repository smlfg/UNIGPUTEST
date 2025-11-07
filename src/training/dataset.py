#!/usr/bin/env python3
"""
Dataset Loading and Preprocessing
Supports HuggingFace datasets and custom formats
"""

from typing import Dict, List, Optional, Any
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


class DatasetLoader:
    """Load and preprocess datasets for fine-tuning"""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        train_split: float = 0.95,
        eval_split: float = 0.05,
    ):
        """
        Initialize dataset loader

        Args:
            dataset_name: HuggingFace dataset name or local path
            tokenizer: Tokenizer for text encoding
            max_seq_length: Maximum sequence length
            train_split: Training data split ratio
            eval_split: Evaluation data split ratio
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_split = train_split
        self.eval_split = eval_split

        self.dataset = None
        self.train_dataset = None
        self.eval_dataset = None

    def load(self) -> Dict[str, Dataset]:
        """
        Load dataset from HuggingFace or local path

        Returns:
            dict: Dictionary with 'train' and 'eval' datasets
        """
        print(f"📦 Loading dataset: {self.dataset_name}")

        try:
            # Try loading from HuggingFace Hub
            self.dataset = load_dataset(self.dataset_name)
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

        # Split dataset if needed
        if isinstance(self.dataset, dict):
            if "train" in self.dataset:
                train_data = self.dataset["train"]
            else:
                # Use first available split
                train_data = list(self.dataset.values())[0]
        else:
            train_data = self.dataset

        # Create train/eval split
        if self.eval_split > 0:
            split_dataset = train_data.train_test_split(
                test_size=self.eval_split,
                seed=42
            )
            self.train_dataset = split_dataset["train"]
            self.eval_dataset = split_dataset["test"]
        else:
            self.train_dataset = train_data
            self.eval_dataset = None

        print(f"✅ Dataset loaded successfully!")
        print(f"   Train samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"   Eval samples: {len(self.eval_dataset)}")

        return {
            "train": self.train_dataset,
            "eval": self.eval_dataset
        }

    def preprocess(
        self,
        prompt_template: Optional[str] = None,
        input_template: Optional[str] = None
    ) -> Dict[str, Dataset]:
        """
        Preprocess dataset with tokenization and formatting

        Args:
            prompt_template: Template for formatting prompts
            input_template: Template for prompts with input field

        Returns:
            dict: Preprocessed datasets
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Default templates (Alpaca format)
        if prompt_template is None:
            prompt_template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}"
            )

        if input_template is None:
            input_template = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
                "### Response:\n{output}"
            )

        def format_prompt(example: Dict[str, Any]) -> str:
            """Format a single example using templates"""
            # Check if example has 'input' field
            if "input" in example and example["input"]:
                template = input_template
            else:
                template = prompt_template

            # Handle different field names
            instruction = example.get("instruction", example.get("prompt", ""))
            output = example.get("output", example.get("response", example.get("completion", "")))
            input_text = example.get("input", "")

            return template.format(
                instruction=instruction,
                output=output,
                input=input_text
            )

        def tokenize_function(examples: Dict[str, List]) -> Dict[str, List]:
            """Tokenize a batch of examples"""
            # Format prompts
            prompts = []
            for i in range(len(examples[list(examples.keys())[0]])):
                example = {k: v[i] for k, v in examples.items()}
                prompts.append(format_prompt(example))

            # Tokenize
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors=None,  # Return lists, not tensors
            )

            # Add labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        print("🔄 Preprocessing dataset...")

        # Process train dataset
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train dataset"
        )

        # Process eval dataset
        if self.eval_dataset:
            self.eval_dataset = self.eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.eval_dataset.column_names,
                desc="Tokenizing eval dataset"
            )

        print("✅ Preprocessing complete!")

        return {
            "train": self.train_dataset,
            "eval": self.eval_dataset
        }


class CustomDatasetLoader(DatasetLoader):
    """Loader for custom dataset formats"""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        train_split: float = 0.95,
        eval_split: float = 0.05,
    ):
        """
        Initialize custom dataset loader

        Args:
            data_path: Path to custom dataset (JSON, CSV, or JSONL)
            tokenizer: Tokenizer for text encoding
            max_seq_length: Maximum sequence length
            train_split: Training data split ratio
            eval_split: Evaluation data split ratio
        """
        super().__init__(
            dataset_name=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            train_split=train_split,
            eval_split=eval_split
        )

    def load(self) -> Dict[str, Dataset]:
        """Load custom dataset from file"""
        import os
        from pathlib import Path

        data_path = Path(self.dataset_name)

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        # Determine file format
        extension = data_path.suffix.lower()

        if extension == ".json":
            self.dataset = load_dataset("json", data_files=str(data_path))
        elif extension == ".jsonl":
            self.dataset = load_dataset("json", data_files=str(data_path))
        elif extension == ".csv":
            self.dataset = load_dataset("csv", data_files=str(data_path))
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        # Continue with standard processing
        return super().load()


if __name__ == "__main__":
    print("Dataset loader module - use in training scripts")
