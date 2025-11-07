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
        """Print training dashboard with modern UI"""
        if clear_screen:
            print("\033[2J\033[H", end="")  # Clear screen

        # ANSI color codes
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        MAGENTA = "\033[95m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        DIM = "\033[2m"

        width = 84

        print()
        print(f"{MAGENTA}╔{'═' * (width - 2)}╗{RESET}")
        print(f"{MAGENTA}║{RESET} {BOLD}🔥 TRAINING DASHBOARD{RESET}{' ' * (width - 24)}║")
        print(f"{MAGENTA}╠{'═' * (width - 2)}╣{RESET}")

        latest = self.get_latest_stats()

        if not latest:
            print(f"{MAGENTA}║{RESET}  {YELLOW}⏳ Waiting for training to start...{RESET}{' ' * (width - 38)}║")
            print(f"{MAGENTA}╚{'═' * (width - 2)}╝{RESET}")
            return

        # Timestamp
        timestamp = latest['timestamp'].strftime('%H:%M:%S')
        print(f"{MAGENTA}║{RESET} {DIM}⏱  {timestamp}{' ' * (width - 14)}║{RESET}")
        print(f"{MAGENTA}╠{'═' * (width - 2)}╣{RESET}")

        # Current stats
        epoch = latest['epoch']
        loss = latest['loss']
        lr = latest['learning_rate']

        # Loss color based on value (lower is better)
        if loss < 0.5:
            loss_color = GREEN
            loss_status = "EXCELLENT"
        elif loss < 1.0:
            loss_color = GREEN
            loss_status = "GOOD"
        elif loss < 1.5:
            loss_color = YELLOW
            loss_status = "FAIR"
        else:
            loss_color = YELLOW
            loss_status = "TRAINING"

        print(f"{MAGENTA}║{RESET}  {BOLD}📊 Current Metrics{RESET}{' ' * (width - 22)}║")
        print(f"{MAGENTA}║{RESET}    Epoch: {CYAN}{epoch:.2f}{RESET} / {CYAN}3.00{RESET}{' ' * (width - 27)}║")
        print(f"{MAGENTA}║{RESET}    Loss:  {loss_color}{loss:.4f} {loss_status}{RESET}{' ' * (width - 29 - len(loss_status))}║")
        print(f"{MAGENTA}║{RESET}    LR:    {DIM}{lr:.6f}{RESET}{' ' * (width - 25)}║")
        print(f"{MAGENTA}╠{'─' * (width - 2)}╣{RESET}")

        # Progress bar
        progress = epoch / 3.0
        bar_width = 68
        filled = int(bar_width * progress)

        # Colored progress bar
        if progress < 0.33:
            bar_color = YELLOW
        elif progress < 0.66:
            bar_color = CYAN
        else:
            bar_color = GREEN

        bar = f"{bar_color}{'▓' * filled}{'░' * (bar_width - filled)}{RESET}"
        progress_str = f"{progress*100:.1f}%"

        print(f"{MAGENTA}║{RESET}  {BOLD}Progress{RESET}{' ' * (width - 12)}║")
        print(f"{MAGENTA}║{RESET}    {bar} {progress_str}{' ' * (width - 78 - len(progress_str))}║")

        # Loss trend
        if len(self.stats_history) >= 10:
            print(f"{MAGENTA}╠{'─' * (width - 2)}╣{RESET}")
            recent_losses = [s['loss'] for s in self.stats_history[-10:]]
            is_decreasing = recent_losses[-1] < recent_losses[0]

            if is_decreasing:
                trend_color = GREEN
                trend_icon = "📉"
                trend_text = "DECREASING ✓"
            else:
                trend_color = RED
                trend_icon = "📈"
                trend_text = "INCREASING ⚠"

            print(f"{MAGENTA}║{RESET}  {BOLD}Loss Trend (last 10 steps){RESET}{' ' * (width - 30)}║")
            print(f"{MAGENTA}║{RESET}    {trend_icon} {trend_color}{trend_text}{RESET}{' ' * (width - 22 - len(trend_text))}║")
            print(f"{MAGENTA}║{RESET}    {recent_losses[0]:.4f} → {recent_losses[-1]:.4f} {DIM}(Δ {recent_losses[-1] - recent_losses[0]:+.4f}){RESET}{' ' * (width - 44)}║")

        # Time estimates
        time_remaining = self.estimate_time_remaining()
        if time_remaining or self.start_time:
            print(f"{MAGENTA}╠{'─' * (width - 2)}╣{RESET}")
            print(f"{MAGENTA}║{RESET}  {BOLD}⏱️  Time{RESET}{' ' * (width - 13)}║")

            if self.start_time:
                elapsed = datetime.now() - self.start_time
                elapsed_str = str(elapsed).split('.')[0]
                print(f"{MAGENTA}║{RESET}    Elapsed:   {CYAN}{elapsed_str}{RESET}{' ' * (width - 26 - len(elapsed_str))}║")

            if time_remaining:
                print(f"{MAGENTA}║{RESET}    Remaining: {YELLOW}{time_remaining}{RESET}{' ' * (width - 26 - len(time_remaining))}║")

        # Stats summary
        if len(self.stats_history) > 1:
            print(f"{MAGENTA}╠{'─' * (width - 2)}╣{RESET}")
            print(f"{MAGENTA}║{RESET}  {BOLD}📊 Statistics{RESET}{' ' * (width - 17)}║")

            steps = len(self.stats_history)
            best_loss = min(s['loss'] for s in self.stats_history)
            worst_loss = max(s['loss'] for s in self.stats_history)
            avg_loss = sum(s['loss'] for s in self.stats_history) / len(self.stats_history)

            print(f"{MAGENTA}║{RESET}    Total Steps:  {CYAN}{steps}{RESET}{' ' * (width - 24 - len(str(steps)))}║")
            print(f"{MAGENTA}║{RESET}    Best Loss:    {GREEN}{best_loss:.4f}{RESET}{' ' * (width - 27)}║")
            print(f"{MAGENTA}║{RESET}    Avg Loss:     {DIM}{avg_loss:.4f}{RESET}{' ' * (width - 27)}║")
            print(f"{MAGENTA}║{RESET}    Worst Loss:   {DIM}{worst_loss:.4f}{RESET}{' ' * (width - 27)}║")

        print(f"{MAGENTA}╚{'═' * (width - 2)}╝{RESET}")
        print()
        print(f"{DIM}  Press Ctrl+C to stop monitoring{RESET}")
        print()


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
