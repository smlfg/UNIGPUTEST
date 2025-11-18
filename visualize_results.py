#!/usr/bin/env python3
"""
Visualisierung der Benchmark-Ergebnisse
=======================================

Erstellt aussagekräftige Plots aus den Benchmark-Daten.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_file: str = "llm_benchmark_results.json"):
    """Lädt Benchmark Results aus JSON"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def plot_memory_comparison(results):
    """Vergleicht GPU Memory Usage"""
    models = []
    fp16_mem = []
    int8_mem = []
    int4_mem = []

    # Gruppiere nach Model
    by_model = {}
    for r in results:
        if not r['success']:
            continue
        model = r['model_name']
        if model not in by_model:
            by_model[model] = {}
        by_model[model][r['quantization']] = r

    # Extrahiere Daten
    for model, quants in by_model.items():
        models.append(model)
        fp16_mem.append(quants.get('FP16', {}).get('memory_gb', 0))
        int8_mem.append(quants.get('8-bit', {}).get('memory_gb', 0))
        int4_mem.append(quants.get('4-bit', {}).get('memory_gb', 0))

    # Plot
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, fp16_mem, width, label='FP16', color='#e74c3c')
    ax.bar(x, int8_mem, width, label='8-bit', color='#f39c12')
    ax.bar(x + width, int4_mem, width, label='4-bit', color='#27ae60')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Memory Usage by Quantization Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: memory_comparison.png")

def plot_throughput_comparison(results):
    """Vergleicht Throughput (tokens/sec)"""
    models = []
    fp16_thr = []
    int8_thr = []
    int4_thr = []

    # Gruppiere nach Model
    by_model = {}
    for r in results:
        if not r['success']:
            continue
        model = r['model_name']
        if model not in by_model:
            by_model[model] = {}
        by_model[model][r['quantization']] = r

    # Extrahiere Daten
    for model, quants in by_model.items():
        models.append(model)
        fp16_thr.append(quants.get('FP16', {}).get('throughput_tokens_per_sec', 0))
        int8_thr.append(quants.get('8-bit', {}).get('throughput_tokens_per_sec', 0))
        int4_thr.append(quants.get('4-bit', {}).get('throughput_tokens_per_sec', 0))

    # Plot
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, fp16_thr, width, label='FP16', color='#e74c3c')
    ax.bar(x, int8_thr, width, label='8-bit', color='#f39c12')
    ax.bar(x + width, int4_thr, width, label='4-bit', color='#27ae60')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Throughput by Quantization Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: throughput_comparison.png")

def plot_latency_comparison(results):
    """Vergleicht First Token Latency"""
    models = []
    fp16_lat = []
    int8_lat = []
    int4_lat = []

    # Gruppiere nach Model
    by_model = {}
    for r in results:
        if not r['success']:
            continue
        model = r['model_name']
        if model not in by_model:
            by_model[model] = {}
        by_model[model][r['quantization']] = r

    # Extrahiere Daten
    for model, quants in by_model.items():
        models.append(model)
        fp16_lat.append(quants.get('FP16', {}).get('first_token_latency_ms', 0))
        int8_lat.append(quants.get('8-bit', {}).get('first_token_latency_ms', 0))
        int4_lat.append(quants.get('4-bit', {}).get('first_token_latency_ms', 0))

    # Plot
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, fp16_lat, width, label='FP16', color='#e74c3c')
    ax.bar(x, int8_lat, width, label='8-bit', color='#f39c12')
    ax.bar(x + width, int4_lat, width, label='4-bit', color='#27ae60')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('First Token Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('First Token Latency by Quantization Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: latency_comparison.png")

def plot_efficiency_heatmap(results):
    """Erstellt Heatmap: Throughput vs Memory"""
    models = []
    quantizations = ['FP16', '8-bit', '4-bit']

    # Daten sammeln
    by_model = {}
    for r in results:
        if not r['success']:
            continue
        model = r['model_name']
        if model not in by_model:
            by_model[model] = {}
        by_model[model][r['quantization']] = r

    # Matrix erstellen
    efficiency_matrix = []
    models = list(by_model.keys())

    for model in models:
        row = []
        for quant in quantizations:
            if quant in by_model[model]:
                r = by_model[model][quant]
                # Efficiency: tokens/sec per GB memory
                efficiency = r['throughput_tokens_per_sec'] / max(r['memory_gb'], 0.01)
                row.append(efficiency)
            else:
                row.append(0)
        efficiency_matrix.append(row)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(efficiency_matrix, cmap='YlGn', aspect='auto')

    # Achsen
    ax.set_xticks(np.arange(len(quantizations)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(quantizations)
    ax.set_yticklabels(models)

    # Werte in Zellen
    for i in range(len(models)):
        for j in range(len(quantizations)):
            text = ax.text(j, i, f'{efficiency_matrix[i][j]:.1f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Efficiency: Tokens/sec per GB Memory', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Efficiency Score')

    plt.tight_layout()
    plt.savefig('efficiency_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: efficiency_heatmap.png")

def main():
    """Hauptprogramm"""
    print("\n" + "="*70)
    print("  Visualizing Benchmark Results")
    print("="*70 + "\n")

    # Check ob Results existieren
    if not Path("llm_benchmark_results.json").exists():
        print("❌ No results file found!")
        print("   Run 'python llm_benchmark.py' first.")
        return

    # Lade Daten
    data = load_results()
    results = data['results']

    print(f"📊 Loaded {len(results)} results")
    print(f"🎮 GPU: {data['metadata']['gpu']}\n")

    # Erstelle Plots
    print("Creating visualizations...")

    plot_memory_comparison(results)
    plot_throughput_comparison(results)
    plot_latency_comparison(results)
    plot_efficiency_heatmap(results)

    print("\n" + "="*70)
    print("✅ All visualizations created!")
    print("="*70 + "\n")

    print("Generated files:")
    print("  - memory_comparison.png")
    print("  - throughput_comparison.png")
    print("  - latency_comparison.png")
    print("  - efficiency_heatmap.png")
    print()

if __name__ == "__main__":
    main()
