#!/bin/bash
# Phase 2: Install Transformers and LLM Tools
# For NVIDIA L40S LLM Development

set -e

echo "========================================"
echo "Phase 2: Transformers & LLM Tools"
echo "========================================"
echo ""

echo "Installing Hugging Face Transformers..."
pip install transformers

echo ""
echo "Installing Accelerate for distributed training..."
pip install accelerate

echo ""
echo "Installing quantization libraries..."
pip install bitsandbytes
pip install auto-gptq

echo ""
echo "Installing tokenizers..."
pip install tokenizers sentencepiece

echo ""
echo "Installing datasets library..."
pip install datasets

echo ""
echo "Installing safetensors for efficient model loading..."
pip install safetensors

echo ""
echo "Installing PEFT for LoRA/QLoRA..."
pip install peft

echo ""
echo "========================================"
echo "Phase 2 Complete!"
echo "========================================"
echo ""
echo "Next step: Test loading a model with test_llm_loading.py"
