import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path

def parse_log(log_path):
    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"Reading log file: {log_path}...")

    pattern = re.compile(
        r'Epoch\s+(\d+)/\d+\s+\|\s+'
        r'Loss:\s+([\d\.]+)\s+\|\s+'
        r'Val Loss:\s+([\d\.]+)\s+\|\s+'
        r'Train Acc:\s+([\d\.]+)\s+\|\s+'
        r'Val Acc:\s+([\d\.]+)'
    )

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))      # Train Loss
                val_loss = float(match.group(3))  # Val Loss
                train_acc = float(match.group(4)) # Train Acc
                val_acc = float(match.group(5))   # Val Acc

                epochs.append(epoch)
                train_losses.append(loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

    print(f"Found {len(epochs)} valid data points.")
    return epochs, train_losses, val_losses, train_accs, val_accs

def plot_metrics(epochs, train_losses, val_losses, train_accs, val_accs, output_dir='.'):
    if not epochs:
        print("Error: No valid data found. Check if the log file contains 'Val Loss'.")
        return

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='tab:blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='tab:orange', linewidth=2)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = Path(output_dir) / 'rnn_loss_plot.png'
    plt.savefig(loss_plot_path, dpi=300)
    print(f"Loss plot saved to: {loss_plot_path}")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='Training Accuracy', color='tab:blue', linewidth=2)
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='tab:orange', linewidth=2)
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    acc_plot_path = Path(output_dir) / 'rnn_acc_plot.png'
    plt.savefig(acc_plot_path, dpi=300)
    print(f"Accuracy plot saved to: {acc_plot_path}")
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from temporal_train_log.txt")
    parser.add_argument('--log', type=str, default='../logs/temporal_train_log.txt', help='Path to the log file')
    parser.add_argument('--out_dir', type=str, default='../samples', help='Directory to save the plots')
    
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        if (Path('logs') / args.log).exists():
            log_path = Path('logs') / args.log
        elif not log_path.exists():
             print(f"Error: Log file not found at {args.log}")
             return

    data = parse_log(log_path)
    plot_metrics(*data, output_dir=args.out_dir)

if __name__ == "__main__":
    main()