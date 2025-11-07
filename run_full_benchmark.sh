#!/bin/bash
# Full Benchmark Suite for LLM Testing
# WARNING: This can take several hours to complete!

set -e

echo "=========================================="
echo "LLM Full Benchmark Suite"
echo "WARNING: This will take several hours!"
echo "=========================================="
echo ""

# Check dependencies
python3 -c "import torch; assert torch.cuda.is_available()" || {
    echo "❌ CUDA not available"
    exit 1
}

echo "✅ Starting full benchmark suite..."
echo ""

# Create output directories
mkdir -p benchmark_results
mkdir -p reports
mkdir -p logs

# Log file
LOG_FILE="logs/benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logging to: $LOG_FILE"

# Function to run benchmark and log
run_benchmark() {
    local model=$1
    local quant=$2
    local batches=$3

    echo ""
    echo "🔄 Testing: $model ($quant)"
    echo "   Batch sizes: $batches"

    python3 llm_benchmark.py \
        --model "$model" \
        --quantization "$quant" \
        --batch-sizes $batches \
        2>&1 | tee -a "$LOG_FILE"
}

# 7B Models - FP16
echo "===== Phase 1: 7B Models (FP16) ====="
run_benchmark "gpt2" "fp16" "1 4 8 16"
run_benchmark "mistralai/Mistral-7B-v0.1" "fp16" "1 2 4 8"
run_benchmark "meta-llama/Llama-2-7b-hf" "fp16" "1 2 4 8"
run_benchmark "codellama/CodeLlama-7b-hf" "fp16" "1 2 4 8"

# 7B Models - INT8
echo "===== Phase 2: 7B Models (INT8) ====="
run_benchmark "mistralai/Mistral-7B-v0.1" "int8" "1 2 4 8 16"
run_benchmark "meta-llama/Llama-2-7b-hf" "int8" "1 2 4 8 16"
run_benchmark "codellama/CodeLlama-7b-hf" "int8" "1 2 4 8 16"

# 13B Models - INT8 (recommended for L40S)
echo "===== Phase 3: 13B Models (INT8) ====="
run_benchmark "meta-llama/Llama-2-13b-hf" "int8" "1 2 4 8"

# 13B Models - INT4
echo "===== Phase 4: 13B Models (INT4) ====="
run_benchmark "meta-llama/Llama-2-13b-hf" "int4" "1 2 4 8 16"

# 34B Model - INT4 (if enough VRAM)
echo "===== Phase 5: 34B Model (INT4) ====="
run_benchmark "codellama/CodeLlama-34b-hf" "int4" "1 2 4 8" || {
    echo "⚠️  34B model skipped (OOM or not available)"
}

# Generate comprehensive reports
echo ""
echo "===== Generating Reports ====="
python3 report_generator.py --format all --plots 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "✅ Full benchmark suite completed!"
echo ""
echo "Results available at:"
echo "  - benchmark_results/"
echo "  - reports/"
echo "  - logs/$LOG_FILE"
echo ""
echo "Open reports/benchmark_report_*.md for detailed analysis"
