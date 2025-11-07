#!/usr/bin/env python3
"""
GPU Monitoring Tool
Real-time GPU usage monitoring during training

Usage:
    python src/utils/gpu_monitor.py

    Or in background:
    python src/utils/gpu_monitor.py --interval 5 --log gpu_usage.csv
"""

import time
import argparse
import csv
from datetime import datetime
from typing import Optional
import torch


def get_gpu_stats() -> dict:
    """
    Get current GPU statistics

    Returns:
        dict: GPU stats including utilization, memory, temperature
    """
    if not torch.cuda.is_available():
        return {
            "timestamp": datetime.now().isoformat(),
            "error": "No GPU available"
        }

    stats = {
        "timestamp": datetime.now().isoformat(),
        "gpu_count": torch.cuda.device_count(),
    }

    # Get stats for GPU 0
    if torch.cuda.device_count() > 0:
        stats["gpu_name"] = torch.cuda.get_device_name(0)

        # Memory stats
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        stats["memory_allocated_gb"] = round(mem_allocated, 2)
        stats["memory_reserved_gb"] = round(mem_reserved, 2)
        stats["memory_total_gb"] = round(mem_total, 2)
        stats["memory_free_gb"] = round(mem_total - mem_reserved, 2)
        stats["memory_usage_percent"] = round(100 * mem_reserved / mem_total, 1)

        # Try to get temperature (not always available in PyTorch)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W

            stats["temperature_c"] = temp
            stats["utilization_percent"] = utilization.gpu
            stats["power_usage_w"] = round(power, 1)
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            stats["temperature_c"] = "N/A"
            stats["utilization_percent"] = "N/A"
            stats["power_usage_w"] = "N/A"

    return stats


def print_gpu_stats(stats: dict, clear_screen: bool = True):
    """
    Pretty print GPU statistics

    Args:
        stats: GPU statistics dictionary
        clear_screen: Whether to clear screen before printing
    """
    if clear_screen:
        print("\033[2J\033[H", end="")  # Clear screen and move cursor to top

    print("=" * 70)
    print("🖥️  GPU MONITORING - Live Stats")
    print("=" * 70)
    print(f"Timestamp: {stats['timestamp']}")
    print()

    if "error" in stats:
        print(f"❌ {stats['error']}")
        return

    print(f"GPU: {stats['gpu_name']}")
    print()

    # Memory bar
    mem_used = stats['memory_reserved_gb']
    mem_total = stats['memory_total_gb']
    mem_percent = stats['memory_usage_percent']

    bar_width = 40
    filled = int(bar_width * mem_percent / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    print("💾 VRAM Usage:")
    print(f"   [{bar}] {mem_percent}%")
    print(f"   Used: {mem_used:.2f} GB / {mem_total:.2f} GB")
    print(f"   Free: {stats['memory_free_gb']:.2f} GB")
    print()

    # Other stats
    if stats['utilization_percent'] != "N/A":
        util = stats['utilization_percent']
        print("⚡ GPU Utilization:")
        util_bar = "█" * int(bar_width * util / 100) + "░" * (bar_width - int(bar_width * util / 100))
        print(f"   [{util_bar}] {util}%")
        print()

    if stats['temperature_c'] != "N/A":
        temp = stats['temperature_c']
        temp_emoji = "🟢" if temp < 60 else "🟡" if temp < 80 else "🔴"
        print(f"🌡️  Temperature: {temp_emoji} {temp}°C")
        print()

    if stats['power_usage_w'] != "N/A":
        power = stats['power_usage_w']
        print(f"⚡ Power Usage: {power}W")
        print()

    print("=" * 70)
    print("Press Ctrl+C to stop monitoring")
    print("=" * 70)


def monitor_gpu(interval: int = 2, log_file: Optional[str] = None):
    """
    Monitor GPU in real-time

    Args:
        interval: Refresh interval in seconds
        log_file: Optional CSV file to log stats
    """
    csv_writer = None
    csv_file = None

    if log_file:
        csv_file = open(log_file, 'w', newline='')
        fieldnames = [
            'timestamp', 'memory_allocated_gb', 'memory_reserved_gb',
            'memory_total_gb', 'memory_free_gb', 'memory_usage_percent',
            'temperature_c', 'utilization_percent', 'power_usage_w'
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        print(f"📝 Logging to: {log_file}")

    try:
        while True:
            stats = get_gpu_stats()
            print_gpu_stats(stats, clear_screen=True)

            if csv_writer and "error" not in stats:
                csv_writer.writerow({k: stats.get(k, 'N/A') for k in csv_writer.fieldnames})
                csv_file.flush()

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        if csv_file:
            csv_file.close()
            print(f"📊 Stats saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="GPU Monitoring Tool")
    parser.add_argument(
        "--interval",
        type=int,
        default=2,
        help="Refresh interval in seconds (default: 2)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="CSV file to log stats (optional)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print stats once and exit"
    )

    args = parser.parse_args()

    if args.once:
        stats = get_gpu_stats()
        print_gpu_stats(stats, clear_screen=False)
    else:
        monitor_gpu(interval=args.interval, log_file=args.log)


if __name__ == "__main__":
    main()
