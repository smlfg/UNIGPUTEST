#!/bin/bash
# Quick Benchmark Script for LLM Testing
# Runs a fast benchmark suite with common models

set -e

echo "=================================="
echo "LLM Quick Benchmark Suite"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
    echo "❌ CUDA not available"
    exit 1
}

echo "✅ CUDA available"
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "✅ GPU: $GPU_NAME"
echo ""

# Create directories
mkdir -p benchmark_results
mkdir -p reports

# Run quick test with GPT-2
echo "🚀 Running quick test with GPT-2..."
python3 llm_benchmark.py --quick

# Generate reports
echo ""
echo "📊 Generating reports..."
python3 report_generator.py --format all --plots

echo ""
echo "✅ Quick benchmark complete!"
echo ""
echo "Results saved to:"
echo "  - benchmark_results/"
echo "  - reports/"
echo ""
echo "Next steps:"
echo "  1. Review reports/benchmark_report_*.md"
echo "  2. Run full suite: python3 llm_benchmark.py"
echo "  3. Monitor GPU: python3 gpu_monitor.py"
