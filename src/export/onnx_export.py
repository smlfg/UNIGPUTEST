#!/usr/bin/env python3
"""
ONNX Export Pipeline
Export PyTorch models to ONNX format with optimization
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig


class ONNXExporter:
    """Export PyTorch models to ONNX format"""

    def __init__(
        self,
        model_path: str,
        output_path: str,
        opset_version: int = 14,
    ):
        """
        Initialize ONNX exporter

        Args:
            model_path: Path to PyTorch model
            output_path: Output path for ONNX model
            opset_version: ONNX opset version
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.opset_version = opset_version

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load PyTorch model and tokenizer"""
        print(f"📦 Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # ONNX export requires float32
            device_map="cpu",  # Export on CPU
        )

        print("✅ Model loaded successfully")

    def export_to_onnx(
        self,
        optimize: bool = True,
        quantize: bool = False,
    ) -> Path:
        """
        Export model to ONNX format

        Args:
            optimize: Apply ONNX optimizations
            quantize: Apply dynamic quantization

        Returns:
            Path: Path to exported ONNX model
        """
        print("🔄 Exporting to ONNX...")
        print(f"   Opset version: {self.opset_version}")
        print(f"   Optimization: {optimize}")
        print(f"   Quantization: {quantize}")

        self.output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Use Optimum for export
            from optimum.onnxruntime import ORTModelForCausalLM

            # Export with optimization
            ort_model = ORTModelForCausalLM.from_pretrained(
                str(self.model_path),
                export=True,
                provider="CPUExecutionProvider",  # Use CPU for export
            )

            # Save ONNX model
            ort_model.save_pretrained(str(self.output_path))
            self.tokenizer.save_pretrained(str(self.output_path))

            print(f"✅ ONNX export complete: {self.output_path}")

        except Exception as e:
            print(f"❌ Export failed: {e}")
            print("   Trying alternative export method...")

            # Fallback: Manual ONNX export
            self._manual_export()

        return self.output_path

    def _manual_export(self):
        """Manual ONNX export (fallback)"""
        import torch.onnx

        if self.model is None:
            self.load_model()

        # Prepare dummy input
        dummy_text = "Hello, world!"
        inputs = self.tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        # Export
        output_file = self.output_path / "model.onnx"

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (inputs["input_ids"], inputs["attention_mask"]),
                str(output_file),
                opset_version=self.opset_version,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence"},
                    "attention_mask": {0: "batch_size", 1: "sequence"},
                    "logits": {0: "batch_size", 1: "sequence"},
                },
            )

        print(f"✅ Manual export complete: {output_file}")

    def optimize_onnx(
        self,
        optimization_level: int = 2,
    ):
        """
        Optimize ONNX model

        Args:
            optimization_level: Optimization level (0-2)
        """
        print(f"⚡ Optimizing ONNX model (level {optimization_level})...")

        try:
            from onnxruntime.transformers import optimizer

            model_file = self.output_path / "model.onnx"

            if not model_file.exists():
                print("❌ ONNX model not found. Export first.")
                return

            optimized_model = optimizer.optimize_model(
                str(model_file),
                model_type="gpt2",  # Generic transformer
                num_heads=0,  # Auto-detect
                hidden_size=0,  # Auto-detect
                optimization_options=None,
            )

            optimized_file = self.output_path / "model_optimized.onnx"
            optimized_model.save_model_to_file(str(optimized_file))

            print(f"✅ Optimization complete: {optimized_file}")

        except Exception as e:
            print(f"⚠️  Optimization failed: {e}")
            print("   Using non-optimized model")

    def quantize_onnx(
        self,
        quantization_type: str = "dynamic",
    ):
        """
        Quantize ONNX model

        Args:
            quantization_type: Type of quantization ('dynamic' or 'static')
        """
        print(f"🔢 Quantizing ONNX model ({quantization_type})...")

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            model_file = self.output_path / "model.onnx"

            if not model_file.exists():
                print("❌ ONNX model not found. Export first.")
                return

            quantized_file = self.output_path / "model_quantized.onnx"

            if quantization_type == "dynamic":
                quantize_dynamic(
                    str(model_file),
                    str(quantized_file),
                    weight_type=QuantType.QUInt8,
                )

            print(f"✅ Quantization complete: {quantized_file}")

            # Print file sizes
            original_size = model_file.stat().st_size / (1024 ** 2)
            quantized_size = quantized_file.stat().st_size / (1024 ** 2)

            print(f"   Original: {original_size:.2f} MB")
            print(f"   Quantized: {quantized_size:.2f} MB")
            print(f"   Compression: {100 * (1 - quantized_size / original_size):.1f}%")

        except Exception as e:
            print(f"❌ Quantization failed: {e}")


def export_model_to_onnx(
    model_path: str,
    output_path: str,
    optimize: bool = True,
    quantize: bool = False,
):
    """
    High-level function to export model to ONNX

    Args:
        model_path: Path to PyTorch model
        output_path: Output path for ONNX model
        optimize: Apply optimizations
        quantize: Apply quantization
    """
    exporter = ONNXExporter(
        model_path=model_path,
        output_path=output_path,
    )

    exporter.load_model()
    exporter.export_to_onnx(optimize=optimize, quantize=quantize)

    if optimize:
        exporter.optimize_onnx()

    if quantize:
        exporter.quantize_onnx()

    print("🎉 Export pipeline complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model-path", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for ONNX")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")
    parser.add_argument("--quantize", action="store_true", help="Quantize ONNX model")

    args = parser.parse_args()

    export_model_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        optimize=args.optimize,
        quantize=args.quantize,
    )
