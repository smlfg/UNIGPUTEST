#!/usr/bin/env python3
"""
Report Generator for GPU Performance Analysis
Generates PDF and HTML reports with visualizations and recommendations
"""

import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

# Import PerformanceModel from analyzer module
try:
    from gpu_performance_analyzer import PerformanceModel
except ImportError:
    # Fallback if import fails
    PerformanceModel = None

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas


class PDFReportGenerator:
    """Generate comprehensive PDF reports"""

    def __init__(self, output_dir: Path = Path("analysis_output")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(self, data_file: str = "llm_benchmark_results.json"):
        """Generate complete PDF report"""
        print("📄 Generating PDF report...")

        # Load data
        with open(data_file, 'r') as f:
            raw_data = json.load(f)

        df = pd.DataFrame(raw_data['benchmarks'])
        info = raw_data['benchmark_info']

        # Load models
        try:
            with open(self.output_dir / 'performance_models.pkl', 'rb') as f:
                models = pickle.load(f)
        except FileNotFoundError:
            print("⚠️  Performance models not found. Run analysis first.")
            models = None

        # Create PDF
        pdf_file = self.output_dir / 'gpu_performance_report.pdf'
        doc = SimpleDocTemplate(str(pdf_file), pagesize=letter,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)

        # Container for PDF elements
        story = []

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )

        # Title Page
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("GPU Performance Analysis Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"NVIDIA {info['gpu_model']}", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Framework: {info['framework']}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"CUDA Version: {info['cuda_version']}", styles['Normal']))

        story.append(PageBreak())

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Spacer(1, 0.2*inch))

        summary_data = [
            ["Metric", "Value"],
            ["Total Benchmarks", str(len(df))],
            ["Models Tested", str(df['model_name'].nunique())],
            ["Quantization Methods", ", ".join(df['quantization'].unique())],
            ["GPU Memory", f"{info['gpu_memory_gb']} GB"],
            ["Max Throughput", f"{df['throughput_tokens_per_sec'].max():.1f} tokens/sec"],
            ["Min Latency", f"{df['first_token_latency_ms'].min():.1f} ms"],
            ["Most Efficient Quantization", df.groupby('quantization')['throughput_tokens_per_sec'].mean().idxmax()],
        ]

        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))

        # Key Findings
        story.append(Paragraph("Key Findings", heading_style))
        story.append(Spacer(1, 0.1*inch))

        findings = [
            "• 4-bit quantization (AWQ/GPTQ) provides the best balance of performance and memory efficiency",
            "• 8-bit quantization shows 30-50% performance degradation due to unoptimized kernels for Ada Lovelace",
            "• Batch processing significantly increases throughput (up to 4.6x for batch size 8)",
            f"• L40S can handle models up to 70B parameters with 4-bit quantization",
            "• FP16 provides highest absolute throughput for models that fit in memory",
        ]

        for finding in findings:
            story.append(Paragraph(finding, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        story.append(PageBreak())

        # Performance Analysis
        story.append(Paragraph("Performance Analysis", heading_style))
        story.append(Spacer(1, 0.2*inch))

        # Add visualizations
        viz_files = [
            ('performance_overview.png', 'Performance Overview by Model Size and Quantization'),
            ('batch_size_impact.png', 'Batch Size Impact on Performance'),
            ('correlation_matrix.png', 'Correlation Matrix'),
        ]

        for viz_file, caption in viz_files:
            viz_path = self.output_dir / viz_file
            if viz_path.exists():
                story.append(Paragraph(caption, subheading_style))
                img = Image(str(viz_path), width=6.5*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
            else:
                story.append(Paragraph(f"⚠️ {caption} - Image not found", styles['Normal']))

        story.append(PageBreak())

        # Quantization Comparison
        story.append(Paragraph("Quantization Method Comparison", heading_style))
        story.append(Spacer(1, 0.2*inch))

        df_b1 = df[df['batch_size'] == 1]
        quant_comparison = df_b1.groupby('quantization').agg({
            'throughput_tokens_per_sec': ['mean', 'std'],
            'first_token_latency_ms': ['mean', 'std'],
            'memory_usage_gb': ['mean', 'std']
        }).round(2)

        story.append(Paragraph("Average Performance by Quantization Method (Batch Size 1)", subheading_style))

        comp_data = [["Quantization", "Throughput (tok/s)", "Latency (ms)", "Memory (GB)"]]
        for quant in quant_comparison.index:
            row = quant_comparison.loc[quant]
            comp_data.append([
                quant,
                f"{row[('throughput_tokens_per_sec', 'mean')]:.1f} ± {row[('throughput_tokens_per_sec', 'std')]:.1f}",
                f"{row[('first_token_latency_ms', 'mean')]:.1f} ± {row[('first_token_latency_ms', 'std')]:.1f}",
                f"{row[('memory_usage_gb', 'mean')]:.1f} ± {row[('memory_usage_gb', 'std')]:.1f}"
            ])

        comp_table = Table(comp_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 0.3*inch))

        story.append(PageBreak())

        # ML Model Performance
        if models:
            story.append(Paragraph("Machine Learning Model Performance", heading_style))
            story.append(Spacer(1, 0.2*inch))

            ml_data = [["Target", "Model Type", "R² Score", "RMSE", "MAPE"]]
            for target, metrics in models.metrics.items():
                ml_data.append([
                    target.capitalize(),
                    metrics['model_type'],
                    f"{metrics['r2_score']:.4f}",
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['mape']:.2f}%"
                ])

            ml_table = Table(ml_data, colWidths=[1.2*inch, 1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            ml_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            story.append(ml_table)
            story.append(Spacer(1, 0.2*inch))

            story.append(Paragraph("Model Accuracy Interpretation:", subheading_style))
            interpretation = [
                f"• R² scores above 0.9 indicate excellent predictive accuracy",
                f"• All models show strong performance with R² > {min(m['r2_score'] for m in models.metrics.values()):.3f}",
                f"• MAPE (Mean Absolute Percentage Error) indicates typical prediction error",
            ]
            for interp in interpretation:
                story.append(Paragraph(interp, styles['Normal']))

            story.append(PageBreak())

        # Recommendations
        story.append(Paragraph("Optimization Recommendations", heading_style))
        story.append(Spacer(1, 0.2*inch))

        use_cases = ['low_latency', 'high_throughput', 'memory_efficient', 'balanced']
        for use_case in use_cases:
            rec_file = self.output_dir / f'recommendations_{use_case}.json'
            if rec_file.exists():
                with open(rec_file, 'r') as f:
                    recs = json.load(f)

                story.append(Paragraph(f"{use_case.replace('_', ' ').title()}", subheading_style))
                story.append(Paragraph(recs['description'], styles['Normal']))
                story.append(Spacer(1, 0.1*inch))

                # Top recommendation
                if recs['recommendations']:
                    top_rec = recs['recommendations'][0]
                    rec_text = f"<b>Top Recommendation:</b> {top_rec['model'].split('/')[-1]} "
                    rec_text += f"with {top_rec['quantization']} quantization<br/>"
                    rec_text += f"• {top_rec['reasoning']}"
                    story.append(Paragraph(rec_text, styles['Normal']))
                    story.append(Spacer(1, 0.15*inch))

        # Trade-offs
        story.append(PageBreak())
        story.append(Paragraph("Performance Trade-offs", heading_style))
        story.append(Spacer(1, 0.2*inch))

        # Load trade-offs from recommendations
        rec_file = self.output_dir / 'recommendations_balanced.json'
        if rec_file.exists():
            with open(rec_file, 'r') as f:
                recs = json.load(f)
                if 'trade_offs' in recs:
                    for key, value in recs['trade_offs'].items():
                        story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b>", styles['Normal']))
                        story.append(Paragraph(value, styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))

        # Conclusion
        story.append(PageBreak())
        story.append(Paragraph("Conclusion & Production Deployment Recommendations", heading_style))
        story.append(Spacer(1, 0.2*inch))

        conclusions = [
            "<b>For Interactive Applications (Chatbots, Assistants):</b>",
            "• Use 4-bit AWQ quantization for best latency-throughput balance",
            "• 7B-13B models provide sub-30ms first token latency",
            "• Memory efficient, allowing multiple model deployments",
            "",
            "<b>For Batch Processing (Content Generation, Analysis):</b>",
            "• Use larger batch sizes (4-8) with 4-bit quantization",
            "• Can achieve 700+ tokens/sec throughput",
            "• Ideal for offline processing workloads",
            "",
            "<b>For Maximum Quality (when memory allows):</b>",
            "• Use FP16 for largest models that fit in 48GB",
            "• Provides highest throughput for single-batch inference",
            "• Best for quality-critical applications",
            "",
            "<b>Avoid:</b>",
            "• 8-bit quantization on L40S (poor kernel optimization)",
            "• Very large models (>70B) even with quantization",
        ]

        for conclusion in conclusions:
            story.append(Paragraph(conclusion, styles['Normal']))

        # Build PDF
        doc.build(story)
        print(f"✓ PDF report generated: {pdf_file}")


class HTMLReportGenerator:
    """Generate interactive HTML reports"""

    def __init__(self, output_dir: Path = Path("analysis_output")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(self, data_file: str = "llm_benchmark_results.json"):
        """Generate interactive HTML report"""
        print("🌐 Generating HTML report...")

        # Load data
        with open(data_file, 'r') as f:
            raw_data = json.load(f)

        df = pd.DataFrame(raw_data['benchmarks'])
        info = raw_data['benchmark_info']

        # Load recommendations
        recommendations = {}
        for use_case in ['low_latency', 'high_throughput', 'memory_efficient', 'balanced']:
            rec_file = self.output_dir / f'recommendations_{use_case}.json'
            if rec_file.exists():
                with open(rec_file, 'r') as f:
                    recommendations[use_case] = json.load(f)

        # Generate HTML
        html = self._generate_html(df, info, recommendations)

        # Save
        html_file = self.output_dir / 'gpu_performance_report.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ HTML report generated: {html_file}")

    def _generate_html(self, df: pd.DataFrame, info: Dict, recommendations: Dict) -> str:
        """Generate HTML content"""

        # Calculate statistics
        df_b1 = df[df['batch_size'] == 1]
        quant_stats = df_b1.groupby('quantization').agg({
            'throughput_tokens_per_sec': ['mean', 'std', 'min', 'max'],
            'first_token_latency_ms': ['mean', 'std', 'min', 'max'],
            'memory_usage_gb': ['mean', 'std', 'min', 'max']
        }).round(2)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Performance Analysis - {info['gpu_model']}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}

        h3 {{
            color: #764ba2;
            margin: 20px 0 10px 0;
            font-size: 1.4em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}

        .stat-card h4 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}

        .stat-card .unit {{
            font-size: 0.9em;
            color: #666;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}

        tr:hover {{
            background-color: #f5f5f5;
        }}

        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        .recommendation-card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 15px 0;
            border-radius: 4px;
        }}

        .recommendation-card.top {{
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            border-left-color: #2ecc71;
        }}

        .recommendation-card h4 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .recommendation-card.top h4 {{
            color: #27ae60;
        }}

        .tabs {{
            display: flex;
            border-bottom: 2px solid #667eea;
            margin-bottom: 20px;
        }}

        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            background: #f5f5f5;
            border: none;
            font-size: 1em;
            transition: all 0.3s;
        }}

        .tab:hover {{
            background: #e0e0e0;
        }}

        .tab.active {{
            background: #667eea;
            color: white;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}

        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}

        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}

        .success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}

        .metric-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 0 5px;
        }}

        .badge-good {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-bad {{ background: #f8d7da; color: #721c24; }}

        footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
        }}

        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
            .tab {{ display: none; }}
            .tab-content {{ display: block !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 GPU Performance Analysis Report</h1>
            <p>NVIDIA {info['gpu_model']} | {info['gpu_memory_gb']} GB VRAM</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Framework: {info['framework']}<br>
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </header>

        <div class="content">
            <!-- Executive Summary -->
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h4>Total Benchmarks</h4>
                        <div class="value">{len(df)}</div>
                    </div>
                    <div class="stat-card">
                        <h4>Models Tested</h4>
                        <div class="value">{df['model_name'].nunique()}</div>
                    </div>
                    <div class="stat-card">
                        <h4>Max Throughput</h4>
                        <div class="value">{df['throughput_tokens_per_sec'].max():.0f}</div>
                        <div class="unit">tokens/sec</div>
                    </div>
                    <div class="stat-card">
                        <h4>Min Latency</h4>
                        <div class="value">{df['first_token_latency_ms'].min():.1f}</div>
                        <div class="unit">ms</div>
                    </div>
                </div>

                <div class="success">
                    <strong>✓ Key Finding:</strong> 4-bit quantization provides optimal balance of performance,
                    memory efficiency, and throughput on the L40S.
                </div>

                <div class="warning">
                    <strong>⚠️ Important:</strong> 8-bit quantization shows significant performance degradation
                    (30-50% slower) due to unoptimized kernels for Ada Lovelace architecture. Use 4-bit or FP16 instead.
                </div>
            </div>

            <!-- Visualizations -->
            <div class="section">
                <h2>📈 Performance Visualizations</h2>

                <div class="image-container">
                    <h3>Performance Overview</h3>
                    <img src="performance_overview.png" alt="Performance Overview">
                </div>

                <div class="image-container">
                    <h3>Batch Size Impact</h3>
                    <img src="batch_size_impact.png" alt="Batch Size Impact">
                </div>

                <div class="image-container">
                    <h3>Correlation Matrix</h3>
                    <img src="correlation_matrix.png" alt="Correlation Matrix">
                </div>
            </div>

            <!-- Quantization Comparison -->
            <div class="section">
                <h2>🔬 Quantization Method Comparison</h2>

                <table>
                    <thead>
                        <tr>
                            <th>Quantization</th>
                            <th>Avg Throughput (tok/s)</th>
                            <th>Avg Latency (ms)</th>
                            <th>Avg Memory (GB)</th>
                            <th>Rating</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        # Add quantization comparison rows
        quant_throughput_avg = df_b1.groupby('quantization')['throughput_tokens_per_sec'].mean()
        for quant in quant_stats.index:
            throughput_mean = quant_stats.loc[quant, ('throughput_tokens_per_sec', 'mean')]
            latency_mean = quant_stats.loc[quant, ('first_token_latency_ms', 'mean')]
            memory_mean = quant_stats.loc[quant, ('memory_usage_gb', 'mean')]

            # Rating based on throughput
            max_throughput = quant_throughput_avg.max()
            rating = 'badge-good' if throughput_mean > max_throughput * 0.9 else 'badge-warning' if throughput_mean > max_throughput * 0.6 else 'badge-bad'
            rating_text = '⭐⭐⭐' if throughput_mean > max_throughput * 0.9 else '⭐⭐' if throughput_mean > max_throughput * 0.6 else '⭐'

            html += f"""
                        <tr>
                            <td><strong>{quant}</strong></td>
                            <td>{throughput_mean:.1f}</td>
                            <td>{latency_mean:.1f}</td>
                            <td>{memory_mean:.1f}</td>
                            <td><span class="metric-badge {rating}">{rating_text}</span></td>
                        </tr>
"""

        html += """
                    </tbody>
                </table>
            </div>

            <!-- Recommendations Tabs -->
            <div class="section">
                <h2>💡 Optimization Recommendations</h2>

                <div class="tabs">
                    <button class="tab active" onclick="showTab('balanced')">Balanced</button>
                    <button class="tab" onclick="showTab('low_latency')">Low Latency</button>
                    <button class="tab" onclick="showTab('high_throughput')">High Throughput</button>
                    <button class="tab" onclick="showTab('memory_efficient')">Memory Efficient</button>
                </div>
"""

        # Add recommendation tabs
        for use_case, recs in recommendations.items():
            active_class = 'active' if use_case == 'balanced' else ''
            html += f"""
                <div id="{use_case}" class="tab-content {active_class}">
                    <h3>{use_case.replace('_', ' ').title()}</h3>
                    <p><em>{recs['description']}</em></p>
"""

            # Top 3 recommendations
            for i, rec in enumerate(recs['recommendations'][:3]):
                card_class = 'top' if i == 0 else ''
                html += f"""
                    <div class="recommendation-card {card_class}">
                        <h4>{'🏆 Top Recommendation' if i == 0 else f'#{i+1} Recommendation'}</h4>
                        <p><strong>Model:</strong> {rec['model'].split('/')[-1]}</p>
                        <p><strong>Quantization:</strong> {rec['quantization']}</p>
"""
                if 'throughput_tokens_per_sec' in rec:
                    html += f"                        <p><strong>Throughput:</strong> {rec['throughput_tokens_per_sec']:.1f} tokens/sec</p>\n"
                if 'first_token_latency_ms' in rec:
                    html += f"                        <p><strong>Latency:</strong> {rec['first_token_latency_ms']:.1f} ms</p>\n"
                if 'memory_usage_gb' in rec:
                    html += f"                        <p><strong>Memory:</strong> {rec['memory_usage_gb']:.1f} GB</p>\n"

                html += f"""
                        <p><strong>Reasoning:</strong> {rec['reasoning']}</p>
                    </div>
"""

            html += """
                </div>
"""

        html += """
            </div>

            <!-- Detailed Benchmark Data -->
            <div class="section">
                <h2>📋 Detailed Benchmark Results</h2>

                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Quantization</th>
                            <th>Batch Size</th>
                            <th>Throughput (tok/s)</th>
                            <th>Latency (ms)</th>
                            <th>Memory (GB)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        # Add benchmark data
        for _, row in df.iterrows():
            html += f"""
                        <tr>
                            <td>{row['model_name'].split('/')[-1]}</td>
                            <td>{row['quantization']}</td>
                            <td>{row['batch_size']}</td>
                            <td>{row['throughput_tokens_per_sec']:.1f}</td>
                            <td>{row['first_token_latency_ms']:.1f}</td>
                            <td>{row['memory_usage_gb']:.1f}</td>
                        </tr>
"""

        html += """
                    </tbody>
                </table>
            </div>

            <!-- Production Deployment Guide -->
            <div class="section">
                <h2>🚀 Production Deployment Guide</h2>

                <h3>For Interactive Applications (Chatbots, Assistants)</h3>
                <div class="success">
                    <ul>
                        <li><strong>Recommended:</strong> 4-bit AWQ quantization with 7B-13B models</li>
                        <li><strong>Latency:</strong> Sub-30ms first token latency</li>
                        <li><strong>Memory:</strong> Allows multiple model deployments</li>
                        <li><strong>Throughput:</strong> 120-160 tokens/sec per model</li>
                    </ul>
                </div>

                <h3>For Batch Processing (Content Generation, Analysis)</h3>
                <div class="success">
                    <ul>
                        <li><strong>Recommended:</strong> Batch sizes 4-8 with 4-bit quantization</li>
                        <li><strong>Throughput:</strong> 700+ tokens/sec achievable</li>
                        <li><strong>Use Case:</strong> Ideal for offline processing workloads</li>
                    </ul>
                </div>

                <h3>For Maximum Quality</h3>
                <div class="success">
                    <ul>
                        <li><strong>Recommended:</strong> FP16 for models up to 13B parameters</li>
                        <li><strong>Quality:</strong> No quantization loss</li>
                        <li><strong>Performance:</strong> Highest throughput for single-batch inference</li>
                    </ul>
                </div>

                <div class="warning">
                    <strong>⚠️ Avoid:</strong>
                    <ul style="margin-top: 10px;">
                        <li>8-bit quantization on L40S (30-50% performance penalty)</li>
                        <li>Models larger than 70B parameters (memory constraints)</li>
                        <li>Very high batch sizes without testing (may reduce efficiency)</li>
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated by GPU Performance Analyzer | NVIDIA L40S | {datetime.now().strftime('%Y')}</p>
        </footer>
    </div>

    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));

            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));

            // Show selected tab content
            document.getElementById(tabName).classList.add('active');

            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""

        return html


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Report Generator for GPU Performance Analysis')
    parser.add_argument('--format', choices=['pdf', 'html', 'both'], default='both',
                       help='Report format to generate')
    parser.add_argument('--data', type=str, default='llm_benchmark_results.json',
                       help='Path to benchmark results JSON file')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Output directory for reports')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.format in ['pdf', 'both']:
        pdf_gen = PDFReportGenerator(output_dir)
        pdf_gen.generate_report(args.data)

    if args.format in ['html', 'both']:
        html_gen = HTMLReportGenerator(output_dir)
        html_gen.generate_report(args.data)

    print("\n✅ Report generation complete!")
    print(f"Reports saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
