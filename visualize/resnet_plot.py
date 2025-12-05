import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def parse_log(log_path):
    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"Reading log file: {log_path}...")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('-') or line.startswith('Training') or line.startswith('Epoch') or line.startswith('TEST'):
            continue
            
        try:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                if parts[0].isdigit():
                    epochs.append(int(parts[0]))
                    train_losses.append(float(parts[1]))
                    val_losses.append(float(parts[2]))
                    train_accs.append(float(parts[3]))
                    val_accs.append(float(parts[4]))
        except ValueError:
            continue

    return epochs, train_losses, val_losses, train_accs, val_accs

def plot_metrics(epochs, train_losses, val_losses, train_accs, val_accs, output_dir='.'):
    if not epochs:
        print("Error: No valid data found in log file.")
        return

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='tab:blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', color='tab:orange', linewidth=2)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = Path(output_dir) / 'resnet_loss_plot.png'
    plt.savefig(loss_plot_path, dpi=300)
    print(f"Loss plot saved to: {loss_plot_path}")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='Train Accuracy', color='tab:blue', linewidth=2)
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='tab:orange', linewidth=2)
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    acc_plot_path = Path(output_dir) / 'resnet_acc_plot.png'
    plt.savefig(acc_plot_path, dpi=300)
    print(f"Accuracy plot saved to: {acc_plot_path}")
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from single_train_log.txt")
    parser.add_argument('--log', type=str, default='../logs/single_train_log.txt', help='Path to the log file')
    parser.add_argument('--out_dir', type=str, default='../samples', help='Directory to save the plots')
    
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        log_path = Path('single_train_log.txt')
        if not log_path.exists():
            print(f"Error: Log file not found at {args.log} or current directory.")
            return

    data = parse_log(log_path)
    plot_metrics(*data, output_dir=args.out_dir)

if __name__ == "__main__":
    main()