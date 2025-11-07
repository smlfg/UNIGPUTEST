"""
LLM Generator
Supports Mixtral 8x22B, Mistral Nemo, and other HuggingFace models
"""

from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMGenerator:
    """
    LLM generator for RAG responses

    Supported models:
    - mistralai/Mixtral-8x22B-Instruct-v0.1 (141B params, 30K context)
    - mistralai/Mistral-Nemo-Instruct-2407 (12B params, 30K context)
    - mistralai/Mistral-7B-Instruct-v0.2 (7B params, 32K context)
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-Nemo-Instruct-2407",
        load_in_4bit: bool = True,
        device_map: str = "auto",
        max_context_length: int = 30000
    ):
        """
        Initialize LLM generator

        Args:
            model_name: HuggingFace model name
            load_in_4bit: Use 4-bit quantization (recommended for large models)
            device_map: Device map ('auto', 'cuda:0', etc.)
            max_context_length: Maximum context length
        """
        self.model_name = model_name
        self.max_context_length = max_context_length

        print(f"Loading LLM: {model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Quantization config for efficient loading
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.model.eval()

        print(f"✅ LLM loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        Generate response from prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Use sampling vs greedy decoding

        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Check context length
        input_length = inputs['input_ids'].shape[1]
        if input_length > self.max_context_length:
            print(f"⚠️  Input length ({input_length}) exceeds max context ({self.max_context_length}), truncating...")
            inputs['input_ids'] = inputs['input_ids'][:, -self.max_context_length:]
            inputs['attention_mask'] = inputs['attention_mask'][:, -self.max_context_length:]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def generate_rag_response(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Generate RAG response given query and retrieved context

        Args:
            query: User query
            context: List of retrieved context chunks
            system_prompt: Optional system prompt
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        from .prompts import RAGPromptBuilder

        # Build prompt
        prompt_builder = RAGPromptBuilder(system_prompt=system_prompt)
        prompt = prompt_builder.build_prompt(query, context)

        # Generate
        response = self.generate(prompt, **generation_kwargs)

        return response

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'max_context_length': self.max_context_length,
            'device': str(self.model.device),
            'dtype': str(self.model.dtype),
        }
