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


def get_colored_bar(percent: float, width: int = 40, style: str = "gradient") -> str:
    """
    Create colored progress bar

    Args:
        percent: Percentage (0-100)
        width: Bar width
        style: 'gradient' or 'solid'

    Returns:
        str: Colored progress bar
    """
    filled = int(width * percent / 100)

    # Color based on percentage
    if percent < 50:
        color = "\033[92m"  # Green
    elif percent < 75:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red

    reset = "\033[0m"

    if style == "gradient":
        # Use different characters for visual effect
        bar_chars = "▓" * filled + "░" * (width - filled)
    else:
        bar_chars = "█" * filled + "░" * (width - filled)

    return f"{color}{bar_chars}{reset}"


def print_gpu_stats(stats: dict, clear_screen: bool = True):
    """
    Pretty print GPU statistics with modern UI

    Args:
        stats: GPU statistics dictionary
        clear_screen: Whether to clear screen before printing
    """
    if clear_screen:
        print("\033[2J\033[H", end="")  # Clear screen and move cursor to top

    # ANSI color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    # Box drawing characters
    width = 76

    print()
    print(f"{CYAN}╔{'═' * (width - 2)}╗{RESET}")
    print(f"{CYAN}║{RESET} {BOLD}🖥️  GPU MONITORING{RESET}{' ' * (width - 20)}║")
    print(f"{CYAN}╠{'═' * (width - 2)}╣{RESET}")

    timestamp = stats['timestamp'].split('T')[1].split('.')[0]
    print(f"{CYAN}║{RESET} {DIM}⏱  {timestamp}{' ' * (width - 14)}║{RESET}")
    print(f"{CYAN}╠{'═' * (width - 2)}╣{RESET}")

    if "error" in stats:
        print(f"{CYAN}║{RESET}  {RED}❌ {stats['error']}{RESET}{' ' * (width - len(stats['error']) - 7)}║")
        print(f"{CYAN}╚{'═' * (width - 2)}╝{RESET}")
        return

    # GPU Name
    gpu_name = stats['gpu_name']
    print(f"{CYAN}║{RESET}  {BOLD}📊 Device:{RESET} {GREEN}{gpu_name}{RESET}{' ' * (width - len(gpu_name) - 15)}║")
    print(f"{CYAN}╠{'═' * (width - 2)}╣{RESET}")

    # VRAM Section
    mem_used = stats['memory_reserved_gb']
    mem_total = stats['memory_total_gb']
    mem_free = stats['memory_free_gb']
    mem_percent = stats['memory_usage_percent']

    print(f"{CYAN}║{RESET}  {BOLD}💾 VRAM{RESET}{' ' * (width - 11)}║")
    bar = get_colored_bar(mem_percent, width=60)
    percent_str = f"{mem_percent:.1f}%"
    print(f"{CYAN}║{RESET}    {bar} {percent_str}{' ' * (width - 70 - len(percent_str))}║")

    usage_str = f"Used: {mem_used:.1f} GB / {mem_total:.1f} GB"
    free_str = f"Free: {mem_free:.1f} GB"
    print(f"{CYAN}║{RESET}    {usage_str}{' ' * (width - len(usage_str) - 6)}║")
    print(f"{CYAN}║{RESET}    {free_str}{' ' * (width - len(free_str) - 6)}║")

    # GPU Utilization
    if stats['utilization_percent'] != "N/A":
        util = stats['utilization_percent']
        print(f"{CYAN}╠{'─' * (width - 2)}╣{RESET}")
        print(f"{CYAN}║{RESET}  {BOLD}⚡ GPU Utilization{RESET}{' ' * (width - 21)}║")

        util_bar = get_colored_bar(util, width=60)
        util_str = f"{util}%"
        print(f"{CYAN}║{RESET}    {util_bar} {util_str}{' ' * (width - 70 - len(util_str))}║")

    # Temperature and Power
    if stats['temperature_c'] != "N/A" or stats['power_usage_w'] != "N/A":
        print(f"{CYAN}╠{'─' * (width - 2)}╣{RESET}")

        info_parts = []

        if stats['temperature_c'] != "N/A":
            temp = stats['temperature_c']
            if temp < 60:
                temp_color = GREEN
                temp_status = "OPTIMAL"
            elif temp < 75:
                temp_color = YELLOW
                temp_status = "WARM"
            elif temp < 85:
                temp_color = YELLOW
                temp_status = "HOT"
            else:
                temp_color = RED
                temp_status = "CRITICAL"

            info_parts.append(f"🌡️  {temp_color}{temp}°C {temp_status}{RESET}")

        if stats['power_usage_w'] != "N/A":
            power = stats['power_usage_w']
            info_parts.append(f"⚡ {power:.1f}W")

        info_line = "    ".join(info_parts)
        # Remove ANSI codes for length calculation
        info_line_plain = info_line.replace(GREEN, "").replace(YELLOW, "").replace(RED, "").replace(RESET, "").replace(BOLD, "").replace(DIM, "")
        print(f"{CYAN}║{RESET}  {info_line}{' ' * (width - len(info_line_plain) - 4)}║")

    print(f"{CYAN}╚{'═' * (width - 2)}╝{RESET}")
    print()
    print(f"{DIM}  Press Ctrl+C to stop monitoring{RESET}")
    print()


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
