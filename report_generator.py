#!/usr/bin/env python3
"""
Automated Report Generator for LLM Benchmarks
Generates JSON, CSV, and Markdown reports with comparisons
"""

import json
import csv
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import glob

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Benchmark result data"""
    model_name: str
    quantization: str
    batch_size: int
    timestamp: str
    tokens_per_second: float
    latency_ms: float
    throughput_tokens: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    memory_peak_gb: float
    memory_fragmentation: float
    gpu_utilization_avg: float
    gpu_utilization_peak: float
    temperature_avg: float
    temperature_peak: float
    power_avg_watts: float
    power_peak_watts: float
    cache_hit_rate: float = 0.0
    model_parameters: int = 0
    device_name: str = ""
    cuda_version: str = ""


class ReportGenerator:
    """Generate comprehensive benchmark reports"""

    def __init__(self, results_dir: str = "benchmark_results", output_dir: str = "reports"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, pattern: str = "*.json") -> List[BenchmarkResult]:
        """Load all benchmark results from JSON files"""
        results = []

        json_files = list(self.results_dir.glob(pattern))
        print(f"📂 Loading {len(json_files)} result files...")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        results.append(BenchmarkResult(**item))
                else:
                    results.append(BenchmarkResult(**data))

            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")

        print(f"✅ Loaded {len(results)} benchmark results")
        return results

    def generate_csv(self, results: List[BenchmarkResult], filename: str = "benchmark_results.csv"):
        """Generate CSV report"""
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='') as f:
            if not results:
                return

            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()

            for result in results:
                writer.writerow(asdict(result))

        print(f"💾 CSV report saved: {output_path}")
        return output_path

    def generate_json(self, results: List[BenchmarkResult], filename: str = "benchmark_results.json"):
        """Generate consolidated JSON report"""
        output_path = self.output_dir / filename

        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_results': len(results),
                'models_tested': len(set(r.model_name for r in results)),
                'quantizations_tested': list(set(r.quantization for r in results)),
            },
            'results': [asdict(r) for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 JSON report saved: {output_path}")
        return output_path

    def generate_markdown(self, results: List[BenchmarkResult], filename: str = "benchmark_report.md"):
        """Generate Markdown report"""
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            # Header
            f.write("# LLM Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Benchmarks:** {len(results)}\n\n")

            if not results:
                f.write("No results found.\n")
                return output_path

            # Executive Summary
            f.write("## Executive Summary\n\n")

            models = set(r.model_name for r in results)
            quants = set(r.quantization for r in results)

            f.write(f"- **Models Tested:** {len(models)}\n")
            f.write(f"- **Quantizations:** {', '.join(sorted(quants))}\n")
            f.write(f"- **GPU:** {results[0].device_name}\n")
            f.write(f"- **CUDA Version:** {results[0].cuda_version}\n\n")

            # Performance Highlights
            f.write("## Performance Highlights\n\n")

            # Best throughput
            best_throughput = max(results, key=lambda r: r.tokens_per_second)
            f.write(f"### 🏆 Highest Throughput\n\n")
            f.write(f"- **Model:** {best_throughput.model_name}\n")
            f.write(f"- **Quantization:** {best_throughput.quantization}\n")
            f.write(f"- **Batch Size:** {best_throughput.batch_size}\n")
            f.write(f"- **Throughput:** {best_throughput.tokens_per_second:.2f} tokens/s\n\n")

            # Best latency
            best_latency = min(results, key=lambda r: r.latency_ms)
            f.write(f"### ⚡ Lowest Latency\n\n")
            f.write(f"- **Model:** {best_latency.model_name}\n")
            f.write(f"- **Quantization:** {best_latency.quantization}\n")
            f.write(f"- **Batch Size:** {best_latency.batch_size}\n")
            f.write(f"- **Latency:** {best_latency.latency_ms:.2f} ms\n\n")

            # Most memory efficient
            best_memory = min(results, key=lambda r: r.memory_peak_gb)
            f.write(f"### 💾 Most Memory Efficient\n\n")
            f.write(f"- **Model:** {best_memory.model_name}\n")
            f.write(f"- **Quantization:** {best_memory.quantization}\n")
            f.write(f"- **Batch Size:** {best_memory.batch_size}\n")
            f.write(f"- **Peak Memory:** {best_memory.memory_peak_gb:.2f} GB\n\n")

            # Most power efficient
            if any(r.power_avg_watts > 0 for r in results):
                power_efficient = min(
                    [r for r in results if r.power_avg_watts > 0],
                    key=lambda r: r.power_avg_watts / r.tokens_per_second if r.tokens_per_second > 0 else float('inf')
                )
                tokens_per_watt = power_efficient.tokens_per_second / power_efficient.power_avg_watts
                f.write(f"### ⚡ Most Power Efficient\n\n")
                f.write(f"- **Model:** {power_efficient.model_name}\n")
                f.write(f"- **Quantization:** {power_efficient.quantization}\n")
                f.write(f"- **Efficiency:** {tokens_per_watt:.2f} tokens/s/W\n\n")

            # Detailed Results by Model
            f.write("## Detailed Results\n\n")

            for model in sorted(models):
                model_results = [r for r in results if r.model_name == model]

                f.write(f"### {model}\n\n")

                # Create comparison table
                f.write("| Quantization | Batch Size | Throughput (tok/s) | Latency (ms) | Memory Peak (GB) | Power (W) | Temp (°C) |\n")
                f.write("|-------------|------------|-------------------|--------------|------------------|-----------|----------|\n")

                for r in sorted(model_results, key=lambda x: (x.quantization, x.batch_size)):
                    f.write(f"| {r.quantization} | {r.batch_size} | "
                           f"{r.tokens_per_second:.2f} | {r.latency_ms:.2f} | "
                           f"{r.memory_peak_gb:.2f} | {r.power_avg_watts:.1f} | "
                           f"{r.temperature_avg:.1f} |\n")

                f.write("\n")

            # Batch Size Analysis
            f.write("## Batch Size Scaling Analysis\n\n")

            for model in sorted(models):
                for quant in sorted(quants):
                    model_quant_results = [
                        r for r in results
                        if r.model_name == model and r.quantization == quant
                    ]

                    if len(model_quant_results) > 1:
                        f.write(f"### {model} ({quant})\n\n")
                        f.write("| Batch Size | Throughput | Scaling Efficiency |\n")
                        f.write("|------------|------------|--------------------|>\n")

                        sorted_results = sorted(model_quant_results, key=lambda x: x.batch_size)
                        baseline_throughput = sorted_results[0].tokens_per_second
                        baseline_batch = sorted_results[0].batch_size

                        for r in sorted_results:
                            expected_throughput = baseline_throughput * (r.batch_size / baseline_batch)
                            efficiency = (r.tokens_per_second / expected_throughput) * 100 if expected_throughput > 0 else 0
                            f.write(f"| {r.batch_size} | {r.tokens_per_second:.2f} tok/s | {efficiency:.1f}% |\n")

                        f.write("\n")

            # Memory Analysis
            f.write("## Memory Usage Analysis\n\n")
            f.write("| Model | Quantization | Batch Size | Allocated | Reserved | Peak | Fragmentation |\n")
            f.write("|-------|--------------|------------|-----------|----------|------|---------------|\n")

            for r in sorted(results, key=lambda x: (x.model_name, x.quantization, x.batch_size)):
                f.write(f"| {r.model_name} | {r.quantization} | {r.batch_size} | "
                       f"{r.memory_allocated_gb:.2f} GB | {r.memory_reserved_gb:.2f} GB | "
                       f"{r.memory_peak_gb:.2f} GB | {r.memory_fragmentation*100:.1f}% |\n")

            f.write("\n")

            # Power and Thermal Analysis
            if any(r.power_avg_watts > 0 for r in results):
                f.write("## Power and Thermal Analysis\n\n")
                f.write("| Model | Quantization | Avg Power (W) | Peak Power (W) | Avg Temp (°C) | Peak Temp (°C) | Efficiency (tok/s/W) |\n")
                f.write("|-------|--------------|---------------|----------------|---------------|----------------|---------------------|\n")

                for r in sorted(results, key=lambda x: (x.model_name, x.quantization)):
                    if r.batch_size == 1:  # Show only batch=1 for clarity
                        efficiency = r.tokens_per_second / r.power_avg_watts if r.power_avg_watts > 0 else 0
                        f.write(f"| {r.model_name} | {r.quantization} | "
                               f"{r.power_avg_watts:.1f} | {r.power_peak_watts:.1f} | "
                               f"{r.temperature_avg:.1f} | {r.temperature_peak:.1f} | "
                               f"{efficiency:.2f} |\n")

                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            f.write("### For Maximum Throughput\n\n")
            throughput_recommendations = sorted(results, key=lambda r: r.tokens_per_second, reverse=True)[:3]
            for i, r in enumerate(throughput_recommendations, 1):
                f.write(f"{i}. **{r.model_name}** ({r.quantization}, batch={r.batch_size}): "
                       f"{r.tokens_per_second:.2f} tok/s\n")

            f.write("\n### For Low Latency\n\n")
            latency_recommendations = sorted(results, key=lambda r: r.latency_ms)[:3]
            for i, r in enumerate(latency_recommendations, 1):
                f.write(f"{i}. **{r.model_name}** ({r.quantization}, batch={r.batch_size}): "
                       f"{r.latency_ms:.2f} ms\n")

            f.write("\n### For Memory Efficiency\n\n")
            memory_recommendations = sorted(results, key=lambda r: r.memory_peak_gb)[:3]
            for i, r in enumerate(memory_recommendations, 1):
                f.write(f"{i}. **{r.model_name}** ({r.quantization}): "
                       f"{r.memory_peak_gb:.2f} GB peak\n")

            if any(r.power_avg_watts > 0 for r in results):
                f.write("\n### For Power Efficiency\n\n")
                power_recs = sorted(
                    [r for r in results if r.power_avg_watts > 0],
                    key=lambda r: r.power_avg_watts / r.tokens_per_second if r.tokens_per_second > 0 else float('inf')
                )[:3]
                for i, r in enumerate(power_recs, 1):
                    efficiency = r.tokens_per_second / r.power_avg_watts if r.power_avg_watts > 0 else 0
                    f.write(f"{i}. **{r.model_name}** ({r.quantization}): "
                           f"{efficiency:.2f} tok/s/W\n")

            f.write("\n---\n")
            f.write(f"\n*Report generated by LLM Benchmark Suite*\n")

        print(f"💾 Markdown report saved: {output_path}")
        return output_path

    def generate_visualizations(self, results: List[BenchmarkResult], prefix: str = "benchmark"):
        """Generate visualization plots"""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️  Visualization libraries not available. Install pandas, matplotlib, and seaborn.")
            return

        if not results:
            print("⚠️  No results to visualize")
            return

        print("📊 Generating visualizations...")

        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])

        # Set style
        sns.set_style("whitegrid")

        # 1. Throughput comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        pivot_data = df.pivot_table(
            values='tokens_per_second',
            index='model_name',
            columns='quantization',
            aggfunc='max'
        )

        pivot_data.plot(kind='bar', ax=ax)
        ax.set_title('Maximum Throughput by Model and Quantization', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Tokens per Second')
        ax.legend(title='Quantization')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_throughput.png", dpi=300)
        plt.close()

        # 2. Memory usage
        fig, ax = plt.subplots(figsize=(12, 6))

        pivot_data = df.pivot_table(
            values='memory_peak_gb',
            index='model_name',
            columns='quantization',
            aggfunc='max'
        )

        pivot_data.plot(kind='bar', ax=ax)
        ax.set_title('Peak Memory Usage by Model and Quantization', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Memory (GB)')
        ax.legend(title='Quantization')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_memory.png", dpi=300)
        plt.close()

        # 3. Batch size scaling
        for model in df['model_name'].unique():
            for quant in df['quantization'].unique():
                model_data = df[(df['model_name'] == model) & (df['quantization'] == quant)]

                if len(model_data) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    ax.plot(model_data['batch_size'], model_data['tokens_per_second'],
                           marker='o', linewidth=2, markersize=8)
                    ax.set_title(f'Batch Size Scaling: {model} ({quant})',
                               fontsize=14, fontweight='bold')
                    ax.set_xlabel('Batch Size')
                    ax.set_ylabel('Throughput (tokens/s)')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()

                    safe_name = model.replace('/', '_')
                    plt.savefig(self.output_dir / f"{prefix}_scaling_{safe_name}_{quant}.png", dpi=300)
                    plt.close()

        # 4. Power efficiency (if available)
        if df['power_avg_watts'].max() > 0:
            df['tokens_per_watt'] = df['tokens_per_second'] / df['power_avg_watts']

            fig, ax = plt.subplots(figsize=(12, 6))

            pivot_data = df.pivot_table(
                values='tokens_per_watt',
                index='model_name',
                columns='quantization',
                aggfunc='max'
            )

            pivot_data.plot(kind='bar', ax=ax)
            ax.set_title('Power Efficiency by Model and Quantization', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel('Tokens per Second per Watt')
            ax.legend(title='Quantization')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{prefix}_power_efficiency.png", dpi=300)
            plt.close()

        print(f"✅ Visualizations saved to {self.output_dir}")

    def compare_sessions(self, session_dirs: List[str], output_name: str = "session_comparison"):
        """Compare results across multiple benchmark sessions"""
        print("📊 Comparing multiple benchmark sessions...")

        all_session_results = {}

        for session_dir in session_dirs:
            session_path = Path(session_dir)
            if session_path.exists():
                self.results_dir = session_path
                results = self.load_results()
                all_session_results[session_path.name] = results

        if not all_session_results:
            print("⚠️  No session results found")
            return

        # Generate comparison report
        output_path = self.output_dir / f"{output_name}.md"

        with open(output_path, 'w') as f:
            f.write("# Benchmark Session Comparison\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Sessions Compared:** {len(all_session_results)}\n\n")

            for session_name, results in all_session_results.items():
                f.write(f"## Session: {session_name}\n\n")
                f.write(f"- **Total Benchmarks:** {len(results)}\n")

                if results:
                    avg_throughput = sum(r.tokens_per_second for r in results) / len(results)
                    avg_memory = sum(r.memory_peak_gb for r in results) / len(results)

                    f.write(f"- **Average Throughput:** {avg_throughput:.2f} tok/s\n")
                    f.write(f"- **Average Memory Peak:** {avg_memory:.2f} GB\n")

                f.write("\n")

        print(f"💾 Session comparison saved: {output_path}")

    def generate_all_reports(self, generate_plots: bool = True):
        """Generate all report formats"""
        print("\n" + "="*70)
        print("📊 Generating Comprehensive Benchmark Reports")
        print("="*70 + "\n")

        results = self.load_results()

        if not results:
            print("⚠️  No results found in {self.results_dir}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.generate_json(results, f"report_{timestamp}.json")
        self.generate_csv(results, f"report_{timestamp}.csv")
        self.generate_markdown(results, f"report_{timestamp}.md")

        if generate_plots:
            self.generate_visualizations(results, f"viz_{timestamp}")

        print(f"\n✅ All reports generated in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Report Generator')
    parser.add_argument('--results-dir', type=str, default='benchmark_results',
                       help='Directory containing benchmark results')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for reports')
    parser.add_argument('--format', choices=['json', 'csv', 'markdown', 'all'],
                       default='all', help='Report format to generate')
    parser.add_argument('--plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--compare-sessions', nargs='+',
                       help='Compare multiple benchmark sessions')

    args = parser.parse_args()

    generator = ReportGenerator(args.results_dir, args.output_dir)

    if args.compare_sessions:
        generator.compare_sessions(args.compare_sessions)
    elif args.format == 'all':
        generator.generate_all_reports(generate_plots=args.plots)
    else:
        results = generator.load_results()

        if args.format == 'json':
            generator.generate_json(results)
        elif args.format == 'csv':
            generator.generate_csv(results)
        elif args.format == 'markdown':
            generator.generate_markdown(results)

        if args.plots:
            generator.generate_visualizations(results)


if __name__ == "__main__":
    main()
