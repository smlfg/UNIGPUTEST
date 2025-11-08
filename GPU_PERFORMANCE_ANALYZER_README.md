# GPU Performance Analyzer & Predictor for NVIDIA L40S

Advanced benchmark analysis and ML-based performance prediction tool for LLM inference on NVIDIA L40S GPU.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange)

## 🎯 Features

### 1. **Comprehensive Data Analysis**
- Loads and analyzes LLM benchmark results
- Calculates correlations between model parameters and performance metrics
- Identifies performance anomalies (e.g., 8-bit quantization inefficiencies)
- Statistical analysis with Z-score based anomaly detection

### 2. **Machine Learning Prediction Models**
- **Throughput Prediction**: Predict tokens/sec based on model parameters
- **Memory Prediction**: Estimate VRAM requirements for new models
- **Latency Prediction**: Forecast first-token latency
- Confidence intervals (95%) for all predictions
- Multiple ML algorithms tested (Ridge, Random Forest, Gradient Boosting)
- Cross-validation for model selection

### 3. **Optimization Recommendations**
- **Low Latency**: Optimized for interactive applications (chatbots, assistants)
- **High Throughput**: Optimized for batch processing and content generation
- **Memory Efficient**: Optimized for running multiple models simultaneously
- **Balanced**: Best overall performance across all metrics
- Trade-off analysis (e.g., "4-bit saves 66% memory with 10% speed gain")

### 4. **Automated Report Generation**
- **PDF Reports**: Professional reports with visualizations and recommendations
- **HTML Reports**: Interactive reports with tabbed navigation
- Publication-ready visualizations (correlation matrices, performance charts)

## 📊 Key Findings from L40S Analysis

### ✅ Recommendations
- **4-bit quantization (AWQ/GPTQ)** provides the best balance of performance and memory
- **FP16** offers highest throughput for models that fit in memory
- **Batch processing** (4-8 samples) can increase throughput up to 4.6x

### ⚠️ Warnings
- **8-bit quantization shows 30-50% performance degradation** on L40S
  - Reason: Unoptimized kernels for Ada Lovelace architecture
  - Alternative: Use 4-bit or FP16 instead

### 📈 Performance Metrics
- **Max Throughput**: 724.6 tokens/sec (Llama-2-7B, 4-bit AWQ, batch=8)
- **Min Latency**: 21.3 ms (GPT-J-6B, 4-bit AWQ)
- **Largest Model**: 70B parameters with 4-bit quantization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd UNIGPUTEST

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
- Python 3.8+
- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- reportlab (for PDF generation)

### Basic Usage

#### 1. Run Full Analysis
```bash
python3 gpu_performance_analyzer.py
```

This will:
- Load benchmark data
- Analyze correlations
- Detect anomalies
- Create visualizations
- Train ML models
- Generate recommendations for all use cases

#### 2. Predict Performance for New Model
```bash
python3 gpu_performance_analyzer.py --predict \
    --model-size 30 \
    --quantization 4bit-awq \
    --batch-size 1
```

Example output:
```
PREDICTION FOR 30.0B MODEL (4bit-awq)
📊 Throughput: 98.9 tokens/sec
   95% CI: [0.0, 245.1]
💾 Memory: 18.3 GB
   95% CI: [11.7, 24.9]
⏱️  First Token Latency: 64.9 ms
   95% CI: [46.9, 83.0]
```

#### 3. Generate Reports
```bash
# Generate both PDF and HTML reports
python3 report_generator.py --format both

# Generate only PDF
python3 report_generator.py --format pdf

# Generate only HTML
python3 report_generator.py --format html
```

## 📁 Project Structure

```
UNIGPUTEST/
├── gpu_performance_analyzer.py    # Main analysis tool
├── report_generator.py             # PDF and HTML report generation
├── llm_benchmark_results.json      # Benchmark data
├── requirements.txt                # Python dependencies
├── analysis_output/                # Generated outputs
│   ├── correlation_matrix.png
│   ├── performance_overview.png
│   ├── batch_size_impact.png
│   ├── performance_models.pkl      # Trained ML models
│   ├── gpu_performance_report.pdf
│   ├── gpu_performance_report.html
│   └── recommendations_*.json
└── GPU_PERFORMANCE_ANALYZER_README.md
```

## 🔧 Advanced Usage

### Custom Data Analysis

```python
from gpu_performance_analyzer import GPUPerformanceAnalyzer

# Initialize analyzer
analyzer = GPUPerformanceAnalyzer(data_path='custom_benchmarks.json')

# Load and analyze data
analyzer.load_data()
analyzer.analyze_correlations()
analyzer.detect_anomalies()
analyzer.visualize_performance()

# Train models
analyzer.train_prediction_models()

# Generate recommendations
analyzer.generate_recommendations(use_case='low_latency')
```

### Programmatic Predictions

```python
from gpu_performance_analyzer import GPUPerformanceAnalyzer

analyzer = GPUPerformanceAnalyzer()
analyzer.load_data()

# Make prediction
prediction = analyzer.predict_performance(
    model_params_b=40,           # 40B parameters
    quantization='4bit-awq',
    batch_size=4,
    context_length=2048
)

print(f"Throughput: {prediction['throughput_tokens_per_sec']['prediction']:.1f} tok/s")
print(f"Memory: {prediction['memory_usage_gb']['prediction']:.1f} GB")
```

## 📊 Benchmark Data Format

The tool expects JSON data in the following format:

```json
{
  "benchmark_info": {
    "gpu_model": "NVIDIA L40S",
    "gpu_memory_gb": 48,
    "cuda_version": "12.1",
    "framework": "vLLM 0.2.7"
  },
  "benchmarks": [
    {
      "model_name": "meta-llama/Llama-2-7b-hf",
      "model_parameters": 7000000000,
      "quantization": "4bit-awq",
      "batch_size": 1,
      "context_length": 2048,
      "throughput_tokens_per_sec": 156.8,
      "first_token_latency_ms": 24.1,
      "memory_usage_gb": 4.8,
      "gpu_utilization_percent": 91
    }
  ]
}
```

## 🎨 Generated Visualizations

### 1. Performance Overview
- Throughput vs Model Size (by quantization)
- Memory Usage vs Model Size
- First Token Latency vs Model Size
- Quantization comparison bar chart

### 2. Batch Size Impact
- Throughput scaling with batch size
- Memory usage scaling with batch size

### 3. Correlation Matrix
- Heatmap showing relationships between all metrics
- Highlights key performance drivers

## 🤖 Machine Learning Models

### Model Performance

| Target | Algorithm | R² Score | RMSE | MAPE |
|--------|-----------|----------|------|------|
| Throughput | Ridge | 0.92 | 74.58 | 36.9% |
| Memory | Ridge | 0.24 | 3.37 | 43.2% |
| Latency | Linear Regression | 0.86 | 9.22 | 17.9% |

### Features Used
- Model size (billions of parameters)
- Quantization method (encoded)
- Batch size
- Context length
- Model architecture (encoded)

### Model Selection
- Multiple algorithms tested per target
- 5-fold cross-validation
- Best model selected based on R² score

## 📈 Use Case Recommendations

### Interactive Applications (Chatbots, Assistants)
```
✓ Quantization: 4-bit AWQ
✓ Model Size: 7B-13B parameters
✓ Expected Latency: <30ms
✓ Memory Usage: 5-9 GB (allows multiple deployments)
```

### Batch Processing (Content Generation)
```
✓ Quantization: 4-bit AWQ
✓ Batch Size: 4-8
✓ Expected Throughput: 700+ tokens/sec
✓ Ideal for offline workloads
```

### Maximum Quality
```
✓ Quantization: FP16
✓ Model Size: Up to 13B (fits in 48GB)
✓ Highest throughput for single-batch
✓ No quantization quality loss
```

## 🔬 Technical Details

### Correlation Analysis
- Pearson correlation coefficients
- Statistical significance testing
- Categorical variable encoding for analysis

### Anomaly Detection
- Z-score based detection (threshold: 2σ)
- Per-quantization performance comparison
- GPU utilization analysis for kernel efficiency

### Prediction Confidence
- 95% confidence intervals using RMSE
- Normal distribution assumption
- Validated against test set

## 📄 Report Contents

### PDF Report Includes:
1. Executive Summary with key metrics
2. Key findings and recommendations
3. Performance visualizations
4. Quantization comparison tables
5. ML model performance metrics
6. Use case specific recommendations
7. Trade-off analysis
8. Production deployment guidelines

### HTML Report Includes:
- All PDF content plus:
- Interactive tabbed navigation
- Responsive design
- Detailed benchmark data table
- Hover effects and visual enhancements

## 🛠️ Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError**
```bash
# Install missing dependencies
pip install pandas numpy scikit-learn matplotlib seaborn reportlab scipy
```

**Issue: Pickle loading error**
```bash
# Ensure report_generator imports PerformanceModel from analyzer
# Already fixed in current version
```

**Issue: Out of memory during analysis**
```bash
# Reduce dataset size or use smaller visualizations
# Modify output resolution in analyzer (dpi parameter)
```

## 🎯 Future Enhancements

- [ ] Support for multi-GPU configurations
- [ ] Real-time monitoring integration
- [ ] Automated benchmark execution
- [ ] Support for additional GPU models (H100, A100)
- [ ] Cost-performance analysis
- [ ] Integration with MLOps platforms

## 📝 Citation

If you use this tool in your research or production, please cite:

```bibtex
@software{gpu_performance_analyzer,
  title = {GPU Performance Analyzer for NVIDIA L40S},
  author = {Your Name},
  year = {2024},
  description = {ML-based performance prediction for LLM inference}
}
```

## 📄 License

MIT License - feel free to use and modify for your needs.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for the ML/AI community**

*Optimizing GPU performance, one model at a time* 🚀
