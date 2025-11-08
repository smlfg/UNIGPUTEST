#!/usr/bin/env python3
"""
GPU Performance Analyzer & Predictor for NVIDIA L40S
Advanced benchmark analysis and ML-based performance prediction
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class BenchmarkData:
    """Container for benchmark data"""
    df: pd.DataFrame
    info: Dict[str, Any]


@dataclass
class PerformanceModel:
    """Container for trained ML models"""
    throughput_model: Any
    memory_model: Any
    latency_model: Any
    scaler: StandardScaler
    label_encoders: Dict[str, LabelEncoder]
    feature_names: List[str]
    metrics: Dict[str, Dict[str, float]]


class GPUPerformanceAnalyzer:
    """Main analyzer class for GPU performance analysis and prediction"""

    def __init__(self, data_path: str = "llm_benchmark_results.json"):
        self.data_path = data_path
        self.data = None
        self.models = None
        self.output_dir = Path("analysis_output")
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self) -> BenchmarkData:
        """Load and parse benchmark data"""
        print("📊 Loading benchmark data...")

        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)

        df = pd.DataFrame(raw_data['benchmarks'])
        info = raw_data['benchmark_info']

        print(f"✓ Loaded {len(df)} benchmark records")
        print(f"  GPU: {info['gpu_model']}")
        print(f"  Models tested: {df['model_name'].nunique()}")
        print(f"  Quantizations: {df['quantization'].unique().tolist()}")

        self.data = BenchmarkData(df=df, info=info)
        return self.data

    def analyze_correlations(self) -> pd.DataFrame:
        """Analyze correlations between features and performance metrics"""
        print("\n🔍 Analyzing correlations...")

        df = self.data.df

        # Encode categorical variables for correlation
        df_encoded = df.copy()
        le_quant = LabelEncoder()
        df_encoded['quantization_encoded'] = le_quant.fit_transform(df['quantization'])

        # Select numeric columns
        numeric_cols = ['model_parameters', 'quantization_encoded', 'batch_size',
                       'context_length', 'throughput_tokens_per_sec',
                       'first_token_latency_ms', 'memory_usage_gb',
                       'gpu_utilization_percent']

        corr_matrix = df_encoded[numeric_cols].corr()

        # Plot correlation heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix: Model Parameters vs Performance Metrics',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Correlation analysis complete")
        print("\nKey findings:")

        # Print interesting correlations
        print(f"  Model size ↔ Throughput: {corr_matrix.loc['model_parameters', 'throughput_tokens_per_sec']:.3f}")
        print(f"  Model size ↔ Memory: {corr_matrix.loc['model_parameters', 'memory_usage_gb']:.3f}")
        print(f"  Model size ↔ Latency: {corr_matrix.loc['model_parameters', 'first_token_latency_ms']:.3f}")
        print(f"  Batch size ↔ Throughput: {corr_matrix.loc['batch_size', 'throughput_tokens_per_sec']:.3f}")

        return corr_matrix

    def detect_anomalies(self) -> pd.DataFrame:
        """Detect performance anomalies, especially for 8-bit quantization"""
        print("\n🚨 Detecting anomalies...")

        df = self.data.df

        # Group by model and quantization
        perf_by_quant = df.groupby(['model_name', 'quantization', 'batch_size']).agg({
            'throughput_tokens_per_sec': 'mean',
            'first_token_latency_ms': 'mean',
            'memory_usage_gb': 'mean',
            'gpu_utilization_percent': 'mean'
        }).reset_index()

        # Analyze 8-bit performance issue
        print("\n🔬 8-bit Quantization Analysis:")

        # Compare quantization methods for same model
        for model in df['model_name'].unique():
            model_data = df[(df['model_name'] == model) & (df['batch_size'] == 1)]
            if len(model_data) > 1:
                print(f"\n  Model: {model.split('/')[-1]}")

                quant_perf = model_data.groupby('quantization').agg({
                    'throughput_tokens_per_sec': 'mean',
                    'memory_usage_gb': 'mean',
                    'gpu_utilization_percent': 'mean'
                }).sort_values('throughput_tokens_per_sec', ascending=False)

                if '8bit' in quant_perf.index:
                    bit8_throughput = quant_perf.loc['8bit', 'throughput_tokens_per_sec']
                    bit8_util = quant_perf.loc['8bit', 'gpu_utilization_percent']
                    max_throughput = quant_perf['throughput_tokens_per_sec'].max()

                    slowdown_pct = ((max_throughput - bit8_throughput) / max_throughput) * 100

                    print(f"    8-bit throughput: {bit8_throughput:.1f} tokens/s")
                    print(f"    Best throughput: {max_throughput:.1f} tokens/s ({quant_perf['throughput_tokens_per_sec'].idxmax()})")
                    print(f"    8-bit slowdown: {slowdown_pct:.1f}%")
                    print(f"    GPU utilization: {bit8_util:.1f}% (indicates kernel inefficiency)")

                    if slowdown_pct > 30:
                        print(f"    ⚠️  ANOMALY: 8-bit is {slowdown_pct:.1f}% slower than optimal!")
                        print(f"    Reason: 8-bit kernels not optimized for Ada Lovelace architecture")

        # Statistical anomaly detection using Z-score
        df_batch1 = df[df['batch_size'] == 1].copy()
        df_batch1['throughput_zscore'] = stats.zscore(df_batch1['throughput_tokens_per_sec'])
        df_batch1['memory_zscore'] = stats.zscore(df_batch1['memory_usage_gb'])

        anomalies = df_batch1[
            (abs(df_batch1['throughput_zscore']) > 2) |
            (abs(df_batch1['memory_zscore']) > 2)
        ]

        if len(anomalies) > 0:
            print(f"\n  📊 Statistical anomalies detected: {len(anomalies)} cases")
            print(anomalies[['model_name', 'quantization', 'throughput_tokens_per_sec',
                           'memory_usage_gb']].to_string(index=False))

        return perf_by_quant

    def visualize_performance(self):
        """Create comprehensive performance visualizations"""
        print("\n📈 Creating visualizations...")

        df = self.data.df

        # 1. Throughput by Model Size and Quantization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Batch size 1 only for fair comparison
        df_b1 = df[df['batch_size'] == 1].copy()
        df_b1['model_size_b'] = df_b1['model_parameters'] / 1e9

        # Throughput vs Model Size
        ax = axes[0, 0]
        for quant in df_b1['quantization'].unique():
            data = df_b1[df_b1['quantization'] == quant]
            ax.scatter(data['model_size_b'], data['throughput_tokens_per_sec'],
                      label=quant, s=100, alpha=0.7)
        ax.set_xlabel('Model Size (Billions of Parameters)', fontweight='bold')
        ax.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
        ax.set_title('Throughput vs Model Size by Quantization', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Memory vs Model Size
        ax = axes[0, 1]
        for quant in df_b1['quantization'].unique():
            data = df_b1[df_b1['quantization'] == quant]
            ax.scatter(data['model_size_b'], data['memory_usage_gb'],
                      label=quant, s=100, alpha=0.7)
        ax.set_xlabel('Model Size (Billions of Parameters)', fontweight='bold')
        ax.set_ylabel('Memory Usage (GB)', fontweight='bold')
        ax.set_title('Memory Usage vs Model Size by Quantization', fontweight='bold')
        ax.axhline(y=48, color='r', linestyle='--', label='L40S Max Memory (48GB)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # First Token Latency vs Model Size
        ax = axes[1, 0]
        for quant in df_b1['quantization'].unique():
            data = df_b1[df_b1['quantization'] == quant]
            ax.scatter(data['model_size_b'], data['first_token_latency_ms'],
                      label=quant, s=100, alpha=0.7)
        ax.set_xlabel('Model Size (Billions of Parameters)', fontweight='bold')
        ax.set_ylabel('First Token Latency (ms)', fontweight='bold')
        ax.set_title('First Token Latency vs Model Size by Quantization', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Quantization Comparison (Bar Chart)
        ax = axes[1, 1]
        quant_avg = df_b1.groupby('quantization')['throughput_tokens_per_sec'].mean().sort_values(ascending=False)
        colors = ['#2ecc71' if q != '8bit' else '#e74c3c' for q in quant_avg.index]
        quant_avg.plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        ax.set_xlabel('Quantization Method', fontweight='bold')
        ax.set_ylabel('Average Throughput (tokens/sec)', fontweight='bold')
        ax.set_title('Average Throughput by Quantization (Batch Size 1)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Batch Size Impact
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Models with multiple batch sizes
        models_with_batches = df.groupby('model_name')['batch_size'].nunique()
        models_with_batches = models_with_batches[models_with_batches > 1].index

        for model in models_with_batches[:3]:  # Top 3 for clarity
            model_data = df[df['model_name'] == model].sort_values('batch_size')
            model_short = model.split('/')[-1][:20]

            axes[0].plot(model_data['batch_size'], model_data['throughput_tokens_per_sec'],
                        marker='o', label=model_short, linewidth=2)
            axes[1].plot(model_data['batch_size'], model_data['memory_usage_gb'],
                        marker='s', label=model_short, linewidth=2)

        axes[0].set_xlabel('Batch Size', fontweight='bold')
        axes[0].set_ylabel('Throughput (tokens/sec)', fontweight='bold')
        axes[0].set_title('Batch Size Impact on Throughput', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Batch Size', fontweight='bold')
        axes[1].set_ylabel('Memory Usage (GB)', fontweight='bold')
        axes[1].set_title('Batch Size Impact on Memory', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_size_impact.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Visualizations saved to:", self.output_dir)

    def train_prediction_models(self) -> PerformanceModel:
        """Train ML models for performance prediction"""
        print("\n🤖 Training prediction models...")

        df = self.data.df.copy()

        # Feature engineering
        df['model_size_b'] = df['model_parameters'] / 1e9

        # Encode categorical variables
        le_quant = LabelEncoder()
        df['quantization_encoded'] = le_quant.fit_transform(df['quantization'])

        le_model = LabelEncoder()
        df['model_encoded'] = le_model.fit_transform(df['model_name'])

        # Feature selection
        feature_cols = ['model_size_b', 'quantization_encoded', 'batch_size',
                       'context_length', 'model_encoded']

        X = df[feature_cols]

        # Target variables
        y_throughput = df['throughput_tokens_per_sec']
        y_memory = df['memory_usage_gb']
        y_latency = df['first_token_latency_ms']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train models for each target
        models = {}
        metrics = {}

        for target_name, y in [('throughput', y_throughput),
                               ('memory', y_memory),
                               ('latency', y_latency)]:
            print(f"\n  Training {target_name} prediction model...")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Try multiple models
            candidates = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            }

            best_score = -np.inf
            best_model = None
            best_model_name = None

            for name, model in candidates.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                           scoring='r2', n_jobs=-1)
                mean_score = cv_scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = name

            # Train best model on full training set
            best_model.fit(X_train, y_train)

            # Evaluate
            y_pred = best_model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            models[target_name] = best_model
            metrics[target_name] = {
                'model_type': best_model_name,
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'cv_score': best_score
            }

            print(f"    Best model: {best_model_name}")
            print(f"    R² Score: {r2:.4f}")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    MAPE: {mape:.2f}%")
            print(f"    CV Score: {best_score:.4f}")

        # Create performance model object
        perf_model = PerformanceModel(
            throughput_model=models['throughput'],
            memory_model=models['memory'],
            latency_model=models['latency'],
            scaler=scaler,
            label_encoders={'quantization': le_quant, 'model': le_model},
            feature_names=feature_cols,
            metrics=metrics
        )

        # Save models
        with open(self.output_dir / 'performance_models.pkl', 'wb') as f:
            pickle.dump(perf_model, f)

        print(f"\n✓ Models saved to: {self.output_dir / 'performance_models.pkl'}")

        self.models = perf_model
        return perf_model

    def predict_performance(self, model_params_b: float, quantization: str,
                          batch_size: int = 1, context_length: int = 2048,
                          model_name: str = None) -> Dict[str, Any]:
        """Predict performance for a new model configuration"""

        if self.models is None:
            print("⚠️  Models not trained yet. Loading from file...")
            with open(self.output_dir / 'performance_models.pkl', 'rb') as f:
                self.models = pickle.load(f)

        # Encode inputs
        try:
            quant_encoded = self.models.label_encoders['quantization'].transform([quantization])[0]
        except ValueError:
            print(f"⚠️  Unknown quantization '{quantization}', using 'fp16'")
            quant_encoded = self.models.label_encoders['quantization'].transform(['fp16'])[0]

        # Use average model encoding if model_name not provided
        if model_name and model_name in self.data.df['model_name'].values:
            model_encoded = self.models.label_encoders['model'].transform([model_name])[0]
        else:
            model_encoded = len(self.models.label_encoders['model'].classes_) // 2

        # Create feature vector
        features = np.array([[model_params_b, quant_encoded, batch_size,
                            context_length, model_encoded]])
        features_scaled = self.models.scaler.transform(features)

        # Predict
        throughput_pred = self.models.throughput_model.predict(features_scaled)[0]
        memory_pred = self.models.memory_model.predict(features_scaled)[0]
        latency_pred = self.models.latency_model.predict(features_scaled)[0]

        # Calculate confidence intervals (approximate)
        throughput_std = self.models.metrics['throughput']['rmse']
        memory_std = self.models.metrics['memory']['rmse']
        latency_std = self.models.metrics['latency']['rmse']

        return {
            'throughput_tokens_per_sec': {
                'prediction': throughput_pred,
                'confidence_interval_95': (
                    max(0, throughput_pred - 1.96 * throughput_std),
                    throughput_pred + 1.96 * throughput_std
                )
            },
            'memory_usage_gb': {
                'prediction': memory_pred,
                'confidence_interval_95': (
                    max(0, memory_pred - 1.96 * memory_std),
                    min(48, memory_pred + 1.96 * memory_std)
                )
            },
            'first_token_latency_ms': {
                'prediction': latency_pred,
                'confidence_interval_95': (
                    max(0, latency_pred - 1.96 * latency_std),
                    latency_pred + 1.96 * latency_std
                )
            },
            'model_metrics': self.models.metrics
        }

    def generate_recommendations(self, use_case: str = 'balanced') -> Dict[str, Any]:
        """Generate optimization recommendations based on use case"""
        print(f"\n💡 Generating recommendations for use case: '{use_case}'...")

        df = self.data.df
        df_b1 = df[df['batch_size'] == 1].copy()

        recommendations = {
            'use_case': use_case,
            'timestamp': datetime.now().isoformat(),
            'recommendations': []
        }

        if use_case == 'low_latency':
            # Minimize first token latency
            best = df_b1.nsmallest(5, 'first_token_latency_ms')

            recommendations['priority'] = 'Minimize first token latency'
            recommendations['description'] = 'Optimized for real-time, interactive applications'

            for _, row in best.iterrows():
                rec = {
                    'model': row['model_name'],
                    'quantization': row['quantization'],
                    'batch_size': row['batch_size'],
                    'first_token_latency_ms': row['first_token_latency_ms'],
                    'throughput_tokens_per_sec': row['throughput_tokens_per_sec'],
                    'memory_usage_gb': row['memory_usage_gb'],
                    'reasoning': f"Lowest latency ({row['first_token_latency_ms']:.1f}ms) with decent throughput"
                }
                recommendations['recommendations'].append(rec)

        elif use_case == 'high_throughput':
            # Maximize throughput
            best = df_b1.nlargest(5, 'throughput_tokens_per_sec')

            recommendations['priority'] = 'Maximize throughput'
            recommendations['description'] = 'Optimized for batch processing and high volume'

            for _, row in best.iterrows():
                rec = {
                    'model': row['model_name'],
                    'quantization': row['quantization'],
                    'batch_size': row['batch_size'],
                    'throughput_tokens_per_sec': row['throughput_tokens_per_sec'],
                    'first_token_latency_ms': row['first_token_latency_ms'],
                    'memory_usage_gb': row['memory_usage_gb'],
                    'reasoning': f"Highest throughput ({row['throughput_tokens_per_sec']:.1f} tok/s)"
                }
                recommendations['recommendations'].append(rec)

        elif use_case == 'memory_efficient':
            # Minimize memory usage while maintaining reasonable performance
            df_efficient = df_b1[df_b1['throughput_tokens_per_sec'] > df_b1['throughput_tokens_per_sec'].median()]
            best = df_efficient.nsmallest(5, 'memory_usage_gb')

            recommendations['priority'] = 'Minimize memory usage'
            recommendations['description'] = 'Optimized for running multiple models or large batch sizes'

            for _, row in best.iterrows():
                memory_saved = 48 - row['memory_usage_gb']
                rec = {
                    'model': row['model_name'],
                    'quantization': row['quantization'],
                    'batch_size': row['batch_size'],
                    'memory_usage_gb': row['memory_usage_gb'],
                    'memory_available_gb': memory_saved,
                    'throughput_tokens_per_sec': row['throughput_tokens_per_sec'],
                    'reasoning': f"Low memory ({row['memory_usage_gb']:.1f}GB), leaving {memory_saved:.1f}GB free"
                }
                recommendations['recommendations'].append(rec)

        else:  # balanced
            # Score based on normalized metrics
            df_scored = df_b1.copy()

            # Normalize metrics (higher is better for all)
            df_scored['throughput_norm'] = (df_scored['throughput_tokens_per_sec'] - df_scored['throughput_tokens_per_sec'].min()) / \
                                          (df_scored['throughput_tokens_per_sec'].max() - df_scored['throughput_tokens_per_sec'].min())

            df_scored['latency_norm'] = 1 - ((df_scored['first_token_latency_ms'] - df_scored['first_token_latency_ms'].min()) / \
                                            (df_scored['first_token_latency_ms'].max() - df_scored['first_token_latency_ms'].min()))

            df_scored['memory_norm'] = 1 - ((df_scored['memory_usage_gb'] - df_scored['memory_usage_gb'].min()) / \
                                           (df_scored['memory_usage_gb'].max() - df_scored['memory_usage_gb'].min()))

            # Weighted score
            df_scored['balanced_score'] = (0.4 * df_scored['throughput_norm'] +
                                          0.3 * df_scored['latency_norm'] +
                                          0.3 * df_scored['memory_norm'])

            best = df_scored.nlargest(5, 'balanced_score')

            recommendations['priority'] = 'Balanced performance'
            recommendations['description'] = 'Optimized for overall best performance across all metrics'

            for _, row in best.iterrows():
                rec = {
                    'model': row['model_name'],
                    'quantization': row['quantization'],
                    'batch_size': row['batch_size'],
                    'balanced_score': row['balanced_score'],
                    'throughput_tokens_per_sec': row['throughput_tokens_per_sec'],
                    'first_token_latency_ms': row['first_token_latency_ms'],
                    'memory_usage_gb': row['memory_usage_gb'],
                    'reasoning': f"Best balanced score ({row['balanced_score']:.3f})"
                }
                recommendations['recommendations'].append(rec)

        # Calculate trade-offs
        recommendations['trade_offs'] = self._calculate_tradeoffs(df_b1)

        # Save recommendations
        with open(self.output_dir / f'recommendations_{use_case}.json', 'w') as f:
            json.dump(recommendations, f, indent=2)

        print(f"✓ Recommendations saved to: {self.output_dir / f'recommendations_{use_case}.json'}")

        return recommendations

    def _calculate_tradeoffs(self, df: pd.DataFrame) -> Dict[str, str]:
        """Calculate performance trade-offs"""
        tradeoffs = {}

        # Memory vs Speed tradeoff
        # Compare 4-bit vs fp16 for same model
        model_groups = df.groupby('model_name')

        memory_savings = []
        speed_changes = []

        for model, group in model_groups:
            if 'fp16' in group['quantization'].values and '4bit-awq' in group['quantization'].values:
                fp16_row = group[group['quantization'] == 'fp16'].iloc[0]
                bit4_row = group[group['quantization'] == '4bit-awq'].iloc[0]

                memory_save_pct = ((fp16_row['memory_usage_gb'] - bit4_row['memory_usage_gb']) /
                                  fp16_row['memory_usage_gb']) * 100
                speed_change_pct = ((bit4_row['throughput_tokens_per_sec'] - fp16_row['throughput_tokens_per_sec']) /
                                   fp16_row['throughput_tokens_per_sec']) * 100

                memory_savings.append(memory_save_pct)
                speed_changes.append(speed_change_pct)

        if memory_savings:
            avg_memory_save = np.mean(memory_savings)
            avg_speed_change = np.mean(speed_changes)

            tradeoffs['4bit_vs_fp16'] = (
                f"4-bit quantization saves ~{avg_memory_save:.0f}% memory and "
                f"{'gains' if avg_speed_change > 0 else 'loses'} ~{abs(avg_speed_change):.0f}% speed vs FP16"
            )

        # 8-bit warning
        tradeoffs['8bit_warning'] = (
            "⚠️ 8-bit quantization shows 30-50% slower performance than 4-bit on L40S "
            "due to unoptimized kernels for Ada Lovelace architecture. Prefer 4-bit or FP16."
        )

        return tradeoffs

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("=" * 80)
        print("🚀 GPU PERFORMANCE ANALYZER - NVIDIA L40S")
        print("=" * 80)

        # Load data
        self.load_data()

        # Analyze correlations
        self.analyze_correlations()

        # Detect anomalies
        self.detect_anomalies()

        # Visualize
        self.visualize_performance()

        # Train models
        self.train_prediction_models()

        # Generate recommendations for all use cases
        for use_case in ['low_latency', 'high_throughput', 'memory_efficient', 'balanced']:
            self.generate_recommendations(use_case)

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("  • correlation_matrix.png")
        print("  • performance_overview.png")
        print("  • batch_size_impact.png")
        print("  • performance_models.pkl")
        print("  • recommendations_*.json")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='GPU Performance Analyzer & Predictor for NVIDIA L40S'
    )
    parser.add_argument('--data', type=str, default='llm_benchmark_results.json',
                       help='Path to benchmark results JSON file')
    parser.add_argument('--predict', action='store_true',
                       help='Run prediction mode')
    parser.add_argument('--model-size', type=float,
                       help='Model size in billions of parameters')
    parser.add_argument('--quantization', type=str, choices=['fp16', '8bit', '4bit-awq', '4bit-gptq'],
                       help='Quantization method')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')

    args = parser.parse_args()

    analyzer = GPUPerformanceAnalyzer(data_path=args.data)

    if args.predict:
        if not args.model_size or not args.quantization:
            print("❌ For prediction mode, --model-size and --quantization are required")
            return

        # Load or train models
        analyzer.load_data()
        try:
            with open(analyzer.output_dir / 'performance_models.pkl', 'rb') as f:
                analyzer.models = pickle.load(f)
        except FileNotFoundError:
            print("Models not found. Training...")
            analyzer.train_prediction_models()

        # Predict
        prediction = analyzer.predict_performance(
            model_params_b=args.model_size,
            quantization=args.quantization,
            batch_size=args.batch_size
        )

        print(f"\n{'='*60}")
        print(f"PREDICTION FOR {args.model_size}B MODEL ({args.quantization})")
        print(f"{'='*60}")
        print(f"\n📊 Throughput: {prediction['throughput_tokens_per_sec']['prediction']:.1f} tokens/sec")
        print(f"   95% CI: [{prediction['throughput_tokens_per_sec']['confidence_interval_95'][0]:.1f}, "
              f"{prediction['throughput_tokens_per_sec']['confidence_interval_95'][1]:.1f}]")
        print(f"\n💾 Memory: {prediction['memory_usage_gb']['prediction']:.1f} GB")
        print(f"   95% CI: [{prediction['memory_usage_gb']['confidence_interval_95'][0]:.1f}, "
              f"{prediction['memory_usage_gb']['confidence_interval_95'][1]:.1f}]")
        print(f"\n⏱️  First Token Latency: {prediction['first_token_latency_ms']['prediction']:.1f} ms")
        print(f"   95% CI: [{prediction['first_token_latency_ms']['confidence_interval_95'][0]:.1f}, "
              f"{prediction['first_token_latency_ms']['confidence_interval_95'][1]:.1f}]")
        print()
    else:
        # Run full analysis
        analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
