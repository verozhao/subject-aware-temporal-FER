
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from landmark_fc_model import LandmarkFCModel

# ------------------------------
# Dataset
# ------------------------------
class LandmarkDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

        feats = []
        for s in self.df['landmark_value'].astype(str):
            vals = [float(x) for x in s.split(',')]
            feats.append(vals)
        self.features = torch.tensor(np.asarray(feats, dtype=np.float32))
        self.targets = torch.tensor([self.label_to_idx[y] for y in self.df['emotion_label']], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ------------------------------
# Train / Eval helpers
# ------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return (loss_sum / max(1, total), correct / max(1, total))

def _plot_dual_curve(values_a: List[float], values_b: List[float], labels: List[str], title: str, ylabel: str, save_path: Path):
    plt.figure()
    epochs = range(1, max(len(values_a), len(values_b)) + 1)
    if len(values_a) > 0:
        plt.plot(list(epochs)[:len(values_a)], values_a, label=labels[0])
    if len(values_b) > 0:
        plt.plot(list(epochs)[:len(values_b)], values_b, label=labels[1])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_one_resolution(df_res: pd.DataFrame, args, save_dir: Path, label_to_idx: Dict[str, int]) -> Dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = LandmarkDataset(df_res, label_to_idx)
    n_total = len(dataset)
    if n_total < 10:
        raise ValueError(f"Not enough samples for split: got {n_total}")

    n_val = max(1, int(n_total * args.val_ratio))
    n_test = max(1, int(n_total * args.test_ratio))
    n_train = max(1, n_total - n_val - n_test)
    if n_train <= 0:
        n_train = max(1, n_total - n_val - n_test)
    splits = [n_train, n_val, n_test]
    while sum(splits) > n_total:
        splits[-1] -= 1
    while sum(splits) < n_total:
        splits[0] += 1

    train_set, val_set, test_set = random_split(dataset, splits, generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    input_dim = dataset.features.shape[1]
    num_classes = len(label_to_idx)

    model = LandmarkFCModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    # histories
    hist = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, val_acc = evaluate(model, val_loader, device)

        # log
        hist["epoch"].append(epoch+1)
        hist["train_loss"].append(float(train_loss))
        hist["train_acc"].append(float(train_acc))
        hist["val_loss"].append(float(val_loss))
        hist["val_acc"].append(float(val_acc))

        if args.verbose:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | loss: {train_loss:.4f} acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if args.early_stop > 0 and patience_counter >= args.early_stop:
                if args.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device)

    # Save model
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"model_{args.resolution_tag}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'label_to_idx': label_to_idx,
    }, ckpt_path)

    # Save history CSV
    hist_df = pd.DataFrame(hist)
    hist_csv = save_dir / f"history_{args.resolution_tag}.csv"
    hist_df.to_csv(hist_csv, index=False)

    # Save combined plots (train vs val)
    loss_png = save_dir / f"loss_{args.resolution_tag}.png"
    acc_png  = save_dir / f"acc_{args.resolution_tag}.png"
    _plot_dual_curve(hist["train_loss"], hist["val_loss"], ["train", "val"], f"Loss ({args.resolution_tag})", "Loss", loss_png)
    _plot_dual_curve(hist["train_acc"],  hist["val_acc"],  ["train", "val"], f"Accuracy ({args.resolution_tag})", "Accuracy", acc_png)

    return {
        'ckpt_path': str(ckpt_path),
        'train_size': len(train_set),
        'val_size': len(val_set),
        'test_size': len(test_set),
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'history_csv': str(hist_csv),
        'plots': [
            str(loss_png),
            str(acc_png),
        ]
    }

# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train three FC models (one per resolution) on landmark features.")
    parser.add_argument("--csv", type=Path, default=Path("rafdb_landmarks_processed_with_blur_scores.csv"), help="Path to the CSV file.")
    parser.add_argument("--save_dir", type=Path, default=Path("landmark_models"), help="Directory to save models.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--early_stop", type=int, default=10, help="Stop if val acc does not improve after N epochs (0 to disable).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"emotion_label", "resolution_level", "landmark_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    labels = sorted(df["emotion_label"].unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    resolutions = ["not_blurry", "medium_blurry", "blurry"]
    results = {}
    for res in resolutions:
        df_res = df[df["resolution_level"] == res].copy()
        if df_res.empty:
            print(f"[WARN] No rows for resolution '{res}', skipping.")
            continue

        args.resolution_tag = res

        print(f"\n=== Training for resolution: {res} (n={len(df_res)}) ===")
        out = train_one_resolution(
            df_res=df_res,
            args=args,
            save_dir=args.save_dir,
            label_to_idx=label_to_idx
        )
        results[res] = out

    summary_path = args.save_dir / "summary.json"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "results": results}, f, indent=2)
    print("\nSaved:", summary_path)

    if results:
        print("\nResolution  |  Val Acc  |  Test Acc  |  Train/Val/Test Sizes  |  Checkpoint")
        print("-" * 90)
        for res, r in results.items():
            sizes = f"{r['train_size']}/{r['val_size']}/{r['test_size']}"
            print(f"{res:12s}  |  {r['best_val_acc']:.4f}  |  {r['test_acc']:.4f}  |  {sizes:>17s}  |  {Path(r['ckpt_path']).name}")
            print(f"  ↳ history: {Path(r['history_csv']).name}")
            for plot in r['plots']:
                print(f"  ↳ plot:    {Path(plot).name}")

if __name__ == "__main__":
    main()
