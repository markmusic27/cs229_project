#!/usr/bin/env python3
"""Plot training and validation loss from training log."""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_training_log(log_path: str) -> list[dict]:
    """Load training log from JSONL file."""
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def plot_loss(log_path: str, output_path: str | None = None, show: bool = True):
    """
    Plot training and validation loss vs epochs.
    
    Args:
        log_path: Path to the training_log.jsonl file
        output_path: Optional path to save the plot image
        show: Whether to display the plot
    """
    entries = load_training_log(log_path)
    
    epochs = [e['epoch'] for e in entries]
    train_losses = [e['train_loss'] for e in entries]
    val_losses = [e['val_loss'] for e in entries]
    
    # Check if we have any valid validation losses
    has_val_loss = any(v is not None for v in val_losses)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training loss
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=1.5, alpha=0.8)
    
    # Plot validation loss if available
    if has_val_loss:
        valid_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
        valid_val_losses = [v for v in val_losses if v is not None]
        ax.plot(valid_epochs, valid_val_losses, 'r-', label='Validation Loss', linewidth=1.5, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss vs Epochs', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot training loss from log file')
    parser.add_argument(
        'log_path',
        type=str,
        nargs='?',
        default='lora_output/training_log.jsonl',
        help='Path to training_log.jsonl file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to save the plot image (e.g., loss_plot.png)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (useful for headless environments)'
    )
    
    args = parser.parse_args()
    
    plot_loss(args.log_path, args.output, show=not args.no_show)


if __name__ == '__main__':
    main()

