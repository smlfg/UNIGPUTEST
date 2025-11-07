#!/usr/bin/env python3
"""
Real-time GPU Monitoring Tool
Displays live GPU statistics during LLM benchmarks
"""

import time
import curses
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class GPUStats:
    """Container for GPU statistics"""

    def __init__(self):
        self.timestamp = None
        self.utilization = 0.0
        self.memory_used = 0.0
        self.memory_total = 0.0
        self.memory_percent = 0.0
        self.temperature = 0.0
        self.power_usage = 0.0
        self.power_limit = 0.0
        self.fan_speed = 0.0
        self.clock_graphics = 0
        self.clock_memory = 0
        self.pcie_rx = 0.0
        self.pcie_tx = 0.0


class RealTimeGPUMonitor:
    """Real-time GPU monitoring with live display"""

    def __init__(self, device_id: int = 0, update_interval: float = 0.5):
        self.device_id = device_id
        self.update_interval = update_interval
        self.history_length = 60  # Keep 60 data points
        self.history = {
            'utilization': deque(maxlen=self.history_length),
            'memory': deque(maxlen=self.history_length),
            'temperature': deque(maxlen=self.history_length),
            'power': deque(maxlen=self.history_length),
        }

        if not PYNVML_AVAILABLE:
            raise ImportError("pynvml not available. Install with: pip install nvidia-ml-py")

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.device_name = pynvml.nvmlDeviceGetName(self.handle)

    def get_stats(self) -> GPUStats:
        """Collect current GPU statistics"""
        stats = GPUStats()
        stats.timestamp = datetime.now()

        try:
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            stats.utilization = util.gpu

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            stats.memory_used = mem_info.used / 1024**3  # GB
            stats.memory_total = mem_info.total / 1024**3  # GB
            stats.memory_percent = (mem_info.used / mem_info.total) * 100

            # Temperature
            stats.temperature = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )

            # Power
            stats.power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Watts
            stats.power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0

            # Fan speed (if available)
            try:
                stats.fan_speed = pynvml.nvmlDeviceGetFanSpeed(self.handle)
            except pynvml.NVMLError:
                stats.fan_speed = 0

            # Clock speeds
            stats.clock_graphics = pynvml.nvmlDeviceGetClockInfo(
                self.handle, pynvml.NVML_CLOCK_GRAPHICS
            )
            stats.clock_memory = pynvml.nvmlDeviceGetClockInfo(
                self.handle, pynvml.NVML_CLOCK_MEM
            )

            # PCIe throughput (if available)
            try:
                pcie = pynvml.nvmlDeviceGetPcieThroughput(
                    self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                stats.pcie_tx = pcie / 1024  # KB/s to MB/s
                pcie = pynvml.nvmlDeviceGetPcieThroughput(
                    self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
                stats.pcie_rx = pcie / 1024  # KB/s to MB/s
            except pynvml.NVMLError:
                stats.pcie_tx = 0
                stats.pcie_rx = 0

        except pynvml.NVMLError as e:
            print(f"Error collecting stats: {e}")

        return stats

    def update_history(self, stats: GPUStats):
        """Update historical data"""
        self.history['utilization'].append(stats.utilization)
        self.history['memory'].append(stats.memory_percent)
        self.history['temperature'].append(stats.temperature)
        self.history['power'].append(stats.power_usage)

    def get_stats_summary(self) -> Dict:
        """Get summary statistics from history"""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[key] = {
                    'current': values[-1],
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                }
        return summary

    def draw_bar(self, value: float, max_value: float, width: int = 40) -> str:
        """Draw a text-based progress bar"""
        filled = int((value / max_value) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"

    def draw_sparkline(self, values: List[float], width: int = 40) -> str:
        """Draw a simple sparkline"""
        if not values or len(values) < 2:
            return ' ' * width

        # Use unicode block elements for sparkline
        blocks = ' ▁▂▃▄▅▆▇█'
        max_val = max(values) if max(values) > 0 else 1
        min_val = min(values)
        val_range = max_val - min_val if max_val != min_val else 1

        # Sample values to fit width
        step = max(1, len(values) // width)
        sampled = [values[i] for i in range(0, len(values), step)][:width]

        sparkline = ''
        for val in sampled:
            normalized = (val - min_val) / val_range
            block_idx = int(normalized * (len(blocks) - 1))
            sparkline += blocks[block_idx]

        return sparkline.ljust(width)

    def display_curses(self, stdscr):
        """Display monitoring interface using curses"""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(int(self.update_interval * 1000))

        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)

        running = True
        while running:
            stdscr.clear()

            # Get current stats
            stats = self.get_stats()
            self.update_history(stats)
            summary = self.get_stats_summary()

            # Header
            stdscr.addstr(0, 0, "=" * 80)
            stdscr.addstr(1, 0, f" GPU Real-Time Monitor - {self.device_name}", curses.A_BOLD)
            stdscr.addstr(2, 0, f" Press 'q' to quit | Update: {self.update_interval}s | {stats.timestamp.strftime('%H:%M:%S')}")
            stdscr.addstr(3, 0, "=" * 80)

            row = 5

            # GPU Utilization
            util_color = curses.color_pair(1) if stats.utilization < 70 else \
                        curses.color_pair(2) if stats.utilization < 90 else curses.color_pair(3)
            stdscr.addstr(row, 0, " GPU Utilization:", curses.A_BOLD)
            stdscr.addstr(row, 20, f"{stats.utilization:5.1f}%", util_color)
            stdscr.addstr(row + 1, 2, self.draw_bar(stats.utilization, 100, 60))
            if 'utilization' in summary:
                stdscr.addstr(row + 2, 2,
                    f"Avg: {summary['utilization']['avg']:.1f}%  "
                    f"Min: {summary['utilization']['min']:.1f}%  "
                    f"Max: {summary['utilization']['max']:.1f}%"
                )
            stdscr.addstr(row + 3, 2, self.draw_sparkline(list(self.history['utilization']), 60))
            row += 5

            # Memory Usage
            mem_color = curses.color_pair(1) if stats.memory_percent < 70 else \
                       curses.color_pair(2) if stats.memory_percent < 90 else curses.color_pair(3)
            stdscr.addstr(row, 0, " Memory Usage:", curses.A_BOLD)
            stdscr.addstr(row, 20, f"{stats.memory_used:5.1f} / {stats.memory_total:.1f} GB ({stats.memory_percent:.1f}%)", mem_color)
            stdscr.addstr(row + 1, 2, self.draw_bar(stats.memory_used, stats.memory_total, 60))
            if 'memory' in summary:
                stdscr.addstr(row + 2, 2,
                    f"Avg: {summary['memory']['avg']:.1f}%  "
                    f"Min: {summary['memory']['min']:.1f}%  "
                    f"Max: {summary['memory']['max']:.1f}%"
                )
            stdscr.addstr(row + 3, 2, self.draw_sparkline(list(self.history['memory']), 60))
            row += 5

            # Temperature
            temp_color = curses.color_pair(1) if stats.temperature < 70 else \
                        curses.color_pair(2) if stats.temperature < 85 else curses.color_pair(3)
            stdscr.addstr(row, 0, " Temperature:", curses.A_BOLD)
            stdscr.addstr(row, 20, f"{stats.temperature:5.1f}°C", temp_color)
            stdscr.addstr(row + 1, 2, self.draw_bar(stats.temperature, 100, 60))
            if 'temperature' in summary:
                stdscr.addstr(row + 2, 2,
                    f"Avg: {summary['temperature']['avg']:.1f}°C  "
                    f"Min: {summary['temperature']['min']:.1f}°C  "
                    f"Max: {summary['temperature']['max']:.1f}°C"
                )
            stdscr.addstr(row + 3, 2, self.draw_sparkline(list(self.history['temperature']), 60))
            row += 5

            # Power Usage
            power_pct = (stats.power_usage / stats.power_limit) * 100 if stats.power_limit > 0 else 0
            power_color = curses.color_pair(1) if power_pct < 70 else \
                         curses.color_pair(2) if power_pct < 90 else curses.color_pair(3)
            stdscr.addstr(row, 0, " Power Usage:", curses.A_BOLD)
            stdscr.addstr(row, 20, f"{stats.power_usage:5.1f} / {stats.power_limit:.1f} W ({power_pct:.1f}%)", power_color)
            stdscr.addstr(row + 1, 2, self.draw_bar(stats.power_usage, stats.power_limit, 60))
            if 'power' in summary:
                stdscr.addstr(row + 2, 2,
                    f"Avg: {summary['power']['avg']:.1f}W  "
                    f"Min: {summary['power']['min']:.1f}W  "
                    f"Max: {summary['power']['max']:.1f}W"
                )
            stdscr.addstr(row + 3, 2, self.draw_sparkline(list(self.history['power']), 60))
            row += 5

            # Additional Info
            stdscr.addstr(row, 0, " Clock Speeds:", curses.A_BOLD)
            stdscr.addstr(row, 20, f"Graphics: {stats.clock_graphics} MHz  Memory: {stats.clock_memory} MHz")
            row += 1

            if stats.fan_speed > 0:
                stdscr.addstr(row, 0, " Fan Speed:", curses.A_BOLD)
                stdscr.addstr(row, 20, f"{stats.fan_speed:.1f}%")
                row += 1

            if stats.pcie_tx > 0 or stats.pcie_rx > 0:
                stdscr.addstr(row, 0, " PCIe Throughput:", curses.A_BOLD)
                stdscr.addstr(row, 20, f"TX: {stats.pcie_tx:.1f} MB/s  RX: {stats.pcie_rx:.1f} MB/s")
                row += 1

            # System info
            if PSUTIL_AVAILABLE:
                row += 1
                cpu_percent = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                stdscr.addstr(row, 0, " System:", curses.A_BOLD)
                stdscr.addstr(row, 20, f"CPU: {cpu_percent:.1f}%  RAM: {mem.percent:.1f}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB)")

            stdscr.refresh()

            # Check for quit command
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                running = False

    def monitor_simple(self, duration: Optional[int] = None):
        """Simple monitoring without curses (for logging)"""
        print(f"Monitoring GPU {self.device_id}: {self.device_name}")
        print("Press Ctrl+C to stop\n")

        start_time = time.time()
        try:
            while True:
                stats = self.get_stats()
                print(f"[{stats.timestamp.strftime('%H:%M:%S')}] "
                      f"GPU: {stats.utilization:5.1f}%  "
                      f"Mem: {stats.memory_used:5.1f}/{stats.memory_total:.1f}GB ({stats.memory_percent:5.1f}%)  "
                      f"Temp: {stats.temperature:5.1f}°C  "
                      f"Power: {stats.power_usage:6.1f}W")

                time.sleep(self.update_interval)

                if duration and (time.time() - start_time) >= duration:
                    break

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

        # Print summary
        summary = self.get_stats_summary()
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for metric, stats in summary.items():
            print(f"{metric.capitalize():15} Avg: {stats['avg']:6.1f}  "
                  f"Min: {stats['min']:6.1f}  Max: {stats['max']:6.1f}")

    def __del__(self):
        """Cleanup"""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='Real-time GPU Monitoring')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--interval', type=float, default=0.5,
                       help='Update interval in seconds (default: 0.5)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Monitoring duration in seconds (default: unlimited)')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple output instead of curses interface')

    args = parser.parse_args()

    if not PYNVML_AVAILABLE:
        print("Error: pynvml not available")
        print("Install with: pip install nvidia-ml-py")
        return

    try:
        monitor = RealTimeGPUMonitor(args.device, args.interval)

        if args.simple:
            monitor.monitor_simple(args.duration)
        else:
            curses.wrapper(monitor.display_curses)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
