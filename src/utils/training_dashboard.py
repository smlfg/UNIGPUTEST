#!/usr/bin/env python3
"""
Training Dashboard
Real-time training progress visualization

Monitors training logs and displays:
- Current epoch and step
- Loss progression
- Learning rate
- Time estimates
- GPU usage

Usage:
    python src/utils/training_dashboard.py --logdir checkpoints/logs
"""

import argparse
import time
import re
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta


class TrainingMonitor:
    """Monitor training progress from logs"""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.stats_history: List[Dict] = []
        self.start_time = None
        self.last_position = 0

    def parse_log_line(self, line: str) -> Optional[Dict]:
        """
        Parse training log line

        Args:
            line: Log line from training

        Returns:
            dict: Parsed stats or None
        """
        # Match lines like: {'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.5}
        pattern = r"\{'loss':\s*([0-9.]+),\s*'learning_rate':\s*([0-9.e-]+),\s*'epoch':\s*([0-9.]+)\}"
        match = re.search(pattern, line)

        if match:
            return {
                'timestamp': datetime.now(),
                'loss': float(match.group(1)),
                'learning_rate': float(match.group(2)),
                'epoch': float(match.group(3)),
            }
        return None

    def read_new_logs(self) -> List[Dict]:
        """Read new log entries since last check"""
        if not self.log_file or not Path(self.log_file).exists():
            return []

        new_stats = []
        with open(self.log_file, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()

        for line in new_lines:
            stats = self.parse_log_line(line)
            if stats:
                new_stats.append(stats)
                self.stats_history.append(stats)

        return new_stats

    def get_latest_stats(self) -> Optional[Dict]:
        """Get latest training stats"""
        if not self.stats_history:
            return None
        return self.stats_history[-1]

    def estimate_time_remaining(self, total_epochs: int = 3) -> Optional[str]:
        """
        Estimate time remaining

        Args:
            total_epochs: Total number of epochs

        Returns:
            str: Estimated time remaining
        """
        if len(self.stats_history) < 2:
            return None

        latest = self.stats_history[-1]
        current_epoch = latest['epoch']

        if current_epoch == 0:
            return None

        # Calculate time per epoch
        if not self.start_time:
            self.start_time = self.stats_history[0]['timestamp']

        elapsed = (datetime.now() - self.start_time).total_seconds()
        time_per_epoch = elapsed / current_epoch

        remaining_epochs = total_epochs - current_epoch
        remaining_seconds = time_per_epoch * remaining_epochs

        return str(timedelta(seconds=int(remaining_seconds)))

    def print_dashboard(self, clear_screen: bool = True):
        """Print training dashboard"""
        if clear_screen:
            print("\033[2J\033[H", end="")  # Clear screen

        print("=" * 80)
        print("🔥 TRAINING DASHBOARD - Live Progress")
        print("=" * 80)
        print()

        latest = self.get_latest_stats()

        if not latest:
            print("⏳ Waiting for training to start...")
            print()
            print("=" * 80)
            return

        # Current stats
        epoch = latest['epoch']
        loss = latest['loss']
        lr = latest['learning_rate']

        print(f"📊 Current Progress:")
        print(f"   Epoch:         {epoch:.2f} / 3.00")
        print(f"   Loss:          {loss:.4f}")
        print(f"   Learning Rate: {lr:.6f}")
        print()

        # Progress bar
        progress = epoch / 3.0
        bar_width = 50
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"Progress: [{bar}] {progress*100:.1f}%")
        print()

        # Loss trend
        if len(self.stats_history) >= 10:
            recent_losses = [s['loss'] for s in self.stats_history[-10:]]
            loss_trend = "📉 Decreasing" if recent_losses[-1] < recent_losses[0] else "📈 Increasing"
            print(f"📈 Loss Trend (last 10 steps): {loss_trend}")
            print(f"   Start: {recent_losses[0]:.4f} → Current: {recent_losses[-1]:.4f}")
            print()

        # Time estimates
        time_remaining = self.estimate_time_remaining()
        if time_remaining:
            print(f"⏱️  Estimated Time Remaining: {time_remaining}")
            print()

        if self.start_time:
            elapsed = datetime.now() - self.start_time
            print(f"⏱️  Elapsed Time: {str(elapsed).split('.')[0]}")
            print()

        # Stats summary
        if len(self.stats_history) > 1:
            print(f"📊 Statistics:")
            print(f"   Total Steps: {len(self.stats_history)}")
            print(f"   Best Loss:   {min(s['loss'] for s in self.stats_history):.4f}")
            print(f"   Worst Loss:  {max(s['loss'] for s in self.stats_history):.4f}")
            print()

        print("=" * 80)
        print(f"Last Update: {latest['timestamp'].strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)


def monitor_training(log_file: str, interval: int = 5):
    """
    Monitor training in real-time

    Args:
        log_file: Path to training log file
        interval: Refresh interval in seconds
    """
    monitor = TrainingMonitor(log_file)

    try:
        while True:
            monitor.read_new_logs()
            monitor.print_dashboard(clear_screen=True)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Training Dashboard")
    parser.add_argument(
        "--log",
        type=str,
        default="training.log",
        help="Path to training log file"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds"
    )

    args = parser.parse_args()

    print(f"📊 Monitoring training from: {args.log}")
    print(f"🔄 Refresh interval: {args.interval}s")
    print()

    monitor_training(args.log, args.interval)


if __name__ == "__main__":
    main()
