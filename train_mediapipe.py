# Run this file instead of the original train.py for 478 landmarks x 3 coordinates
import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from landmark_fc_model import LandmarkFCModel

class MediapipeLandmarkDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: Dict[str, int], use_3d: bool = True):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        feats = []
        for s in self.df['landmark_value'].astype(str):
            vals = [float(x) for x in s.split(',')]
            feats.append(vals)
        self.features = torch.tensor(np.asarray(feats, dtype=np.float32))
        if not use_3d:
            n_landmarks = self.features.shape[1] // 3
            xy_indices = []
            for i in range(n_landmarks):
                xy_indices.extend([i*3, i*3+1])
            self.features = self.features[:, xy_indices]
        self.targets = torch.tensor([self.label_to_idx[y] for y in self.df['emotion_label']], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    df = pd.read_csv(args.csv)
    labels = sorted(df["emotion_label"].unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    if args.resolution:
        df = df[df["resolution_level"] == args.resolution]
    
    dataset = MediapipeLandmarkDataset(df, label_to_idx, use_3d=args.use_3d)
    n = len(dataset)
    n_val, n_test = max(1, int(n * 0.1)), max(1, int(n * 0.1))
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], 
                                                  generator=torch.Generator().manual_seed(args.seed))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    input_dim = dataset.features.shape[1]
    model = LandmarkFCModel(input_dim, len(label_to_idx), tuple(args.hidden_dims), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc, best_state, patience = -1, None, 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        val_loss, val_acc = evaluate(model, val_loader, device)
        if args.verbose:
            print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if args.early_stop > 0 and patience >= args.early_stop:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'input_dim': input_dim, 
                'num_classes': len(label_to_idx), 'label_to_idx': label_to_idx}, save_dir / "model.pt")
    print(f"Val: {best_val_acc:.4f}, Test: {test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("rafdb_mediapipe_landmarks.csv"))
    parser.add_argument("--save_dir", type=Path, default=Path("mediapipe_models"))
    parser.add_argument("--resolution", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--use_3d", action="store_true", help="Use x,y,z coords instead of just x,y")
    train(parser.parse_args())

if __name__ == "__main__":
    main()