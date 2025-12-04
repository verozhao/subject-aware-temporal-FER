import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import copy
from tqdm import tqdm
import re
from landmark_features import extract_features
from model import HybridRNN, LandmarkClassifier


class TemporalFERDataset(Dataset):
    def __init__(self, csv_path, split, sequence_length=10, label_to_idx=None, augment=False):
        self.seq_len = sequence_length
        self.augment = augment
        
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        
        self.raw_label_to_idx = label_to_idx
        self.norm_label_to_idx = {k.lower(): v for k, v in label_to_idx.items()}
        
        if split == 'train':
            print(f"DEBUG: Normalized Label Map: {self.norm_label_to_idx}")
            unique_csv_labels = df['emotion_label'].unique()
            print(f"DEBUG: CSV Labels found: {unique_csv_labels}")
        
        video_groups = {}
        
        missed_labels = set()

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {split} data"):
            fname = row['picture_id']
            raw_label = row['emotion_label']
            
            if raw_label in self.raw_label_to_idx:
                final_label_idx = self.raw_label_to_idx[raw_label]
            elif raw_label.lower() in self.norm_label_to_idx:
                final_label_idx = self.norm_label_to_idx[raw_label.lower()]
            else:
                missed_labels.add(raw_label)
                continue

            if '_frame' in fname:
                parts = fname.split('_frame')
                vid_id = parts[0]
                frame_idx = int(re.search(r'\d+', parts[1]).group())
            else:
                continue
                
            if vid_id not in video_groups:
                video_groups[vid_id] = []
            
            feat_vec = extract_features(row['landmark_value'])
            if np.isnan(feat_vec).any(): continue
                
            video_groups[vid_id].append({
                'frame_idx': frame_idx,
                'feature': feat_vec,
                'label_idx': final_label_idx
            })
            
        if split == 'train' and len(missed_labels) > 0:
            print(f"WARNING: These labels were dropped because they don't match the model: {missed_labels}")

        self.samples = [] 
        for vid_id, frames in video_groups.items():
            frames.sort(key=lambda x: x['frame_idx'])
            if len(frames) < self.seq_len: continue
            
            for i in range(len(frames) - self.seq_len + 1):
                window_frames = frames[i : i + self.seq_len]
                seq_feats = np.array([f['feature'] for f in window_frames])
                
                label_idx = window_frames[-1]['label_idx']
                self.samples.append((seq_feats, label_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        x = torch.tensor(feats, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        if self.augment:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        return x, y


def compute_dataset_stats(dataset):
    print("Computing statistics for the new dataset (RAVDESS)...")
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_feats = []
    for x, y in loader:
        # x shape: [Batch, Seq, 125] -> Flatten to [Batch * Seq, 125]
        all_feats.append(x.view(-1, x.size(-1)))
        
    all_feats = torch.cat(all_feats, dim=0)
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0) + 1e-6
    
    print(f"  > New Mean (first 5): {mean[:5].numpy()}")
    print(f"  > New Std  (first 5): {std[:5].numpy()}")
    return mean, std

def train_temporal_model():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CSV_PATH = 'landmark_features/rafdess_landmarks_split.csv'
    PRETRAINED_PATH = 'geometric_models/model.pt'
    SAVE_DIR = Path('temporal_models')
    SAVE_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 64
    EPOCHS = 100
    SEQ_LEN = 10
    
    print(f"Using device: {DEVICE}")
    print("Loading pretrained single-frame model...")
    ckpt = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    
    old_model = LandmarkClassifier(
        input_dim=ckpt['input_dim'],
        num_classes=ckpt['num_classes'],
        hidden_dims=[256, 256, 256, 128] 
    )
    old_model.load_state_dict(ckpt['model_state_dict'])
    
    label_map = ckpt['label_to_idx']
    
    train_dataset = TemporalFERDataset(CSV_PATH, 'train', SEQ_LEN, label_to_idx=label_map, augment=True)
    val_dataset = TemporalFERDataset(CSV_PATH, 'val', SEQ_LEN, label_to_idx=label_map, augment=False)
    
    if len(train_dataset) == 0:
        print("Error: No training samples created.")
        return

    MEAN, STD = compute_dataset_stats(train_dataset)
    MEAN = MEAN.to(DEVICE)
    STD = STD.to(DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    model = HybridRNN(old_model, hidden_rnn_size=128, num_classes=len(label_map)).to(DEVICE)
    
    optimizer = optim.AdamW([
        {'params': model.encoder_proj.parameters(), 'lr': 1e-4},
        {'params': model.encoder_layers.parameters(), 'lr': 1e-4}, 
        {'params': model.lstm.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-2)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = (x - MEAN.view(1, 1, -1)) / STD.view(1, 1, -1)
            if epoch == 0 and i == 0:
                print(f"DEBUG Check Input Range: Min={x.min().item():.4f}, Max={x.max().item():.4f}")

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation Logic
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = (x - MEAN.view(1, 1, -1)) / STD.view(1, 1, -1)
                logits = model(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = SAVE_DIR / "lstm_model_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': MEAN.cpu(),
                'std': STD.cpu(),
                'label_to_idx': label_map,
                'input_dim': 125,
                'seq_len': SEQ_LEN,
                'val_acc': best_acc
            }, save_path)
            print(f"  >>> Model saved to {save_path}")

if __name__ == "__main__":
    train_temporal_model()