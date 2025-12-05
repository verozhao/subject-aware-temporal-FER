import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path

def parse_log(log_path):
    epochs = []
    d_losses = []
    div_losses = []
    perc_losses = []

    pattern = re.compile(r'Epoch\s+(\d+)/\d+.*D=([-\d\.]+),\s+Div=([-\d\.]+),\s+Perc=([-\d\.]+)')

    print(f"Reading log file: {log_path}...")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                d_losses.append(float(match.group(2)))
                div_losses.append(float(match.group(3)))
                perc_losses.append(float(match.group(4)))

    return epochs, d_losses, div_losses, perc_losses

def plot_losses(epochs, d_losses, div_losses, perc_losses, save_path=None):
    if not epochs:
        print("Error: No data found in log file.")
        return

    # 设置风格
    plt.style.use('ggplot')
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    line1, = ax1.plot(epochs, d_losses, label='Discriminator Loss', color='tab:blue', linewidth=2, linestyle='-')
    line2, = ax1.plot(epochs, perc_losses, label='Generator Loss', color='tab:orange', linewidth=2, linestyle='-')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Discriminator / Generator Loss', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    line3, = ax2.plot(epochs, div_losses, label='Diversity Loss', color='tab:green', linewidth=2, linestyle='--')
    
    ax2.set_ylabel('Diversity Loss (Negative)', fontsize=12, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.grid(False) 

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10, frameon=True, shadow=True)

    plt.title('FaceID-GAN Training Losses', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='../logs/faceid_train_log.txt', help='Path to log file')
    parser.add_argument('--out', type=str, default='../samples/faceid_loss_plot.png', help='Output image path')
    
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    data = parse_log(log_path)
    plot_losses(*data, save_path=args.out)

if __name__ == "__main__":
    main()