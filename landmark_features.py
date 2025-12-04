import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
from pathlib import Path
import argparse

# key Mediapipe landmark indices for facial regions
LANDMARK_GROUPS = {
    'left_eye': [33, 133, 160, 159, 158, 144, 145, 153],
    'right_eye': [362, 263, 387, 386, 385, 373, 374, 380],
    'left_eyebrow': [70, 63, 105, 66, 107],
    'right_eyebrow': [300, 293, 334, 296, 336],
    'nose': [1, 2, 98, 327, 4, 5, 195, 197],
    'upper_lip': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    'lower_lip': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'mouth_corners': [61, 291],
    'jaw': [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
}

KEY_LANDMARKS = [
    1,    # nose tip
    4,    # nose bottom
    33,   # left eye inner
    133,  # left eye outer
    362,  # right eye inner
    263,  # right eye outer
    61,   # left mouth corner
    291,  # right mouth corner
    17,   # lower lip center
    0,    # upper lip center
    152,  # chin
    70,   # left eyebrow inner
    105,  # left eyebrow outer
    300,  # right eyebrow inner
    334,  # right eyebrow outer
    168,  # between eyes
    6,    # nose bridge
    197,  # nose left
    195,  # nose right
    57,   # left cheek
    287,  # right cheek
]

DISTANCE_PAIRS = [
    (33, 133), (362, 263),  # eye widths
    (159, 145), (386, 374),  # eye heights
    (70, 105), (300, 334),   # eyebrow lengths
    (61, 291),               # mouth width
    (0, 17),                 # mouth height (lip distance)
    (33, 362),               # inter-eye distance
    (1, 152),                # nose to chin
    (168, 1),                # between eyes to nose
    (70, 33), (300, 362),    # eyebrow to eye
    (1, 61), (1, 291),       # nose to mouth corners
    (0, 1),                  # upper lip to nose
    (159, 70), (386, 300),   # eye top to eyebrow
    (61, 152), (291, 152),   # mouth corners to chin
    (33, 61), (362, 291),    # eye to mouth corner (same side)
]


def parse_landmarks(landmark_str: str) -> np.ndarray:
    vals = [float(x) for x in landmark_str.split(',')]
    n_coords = len(vals)
    if n_coords % 3 == 0:
        return np.array(vals).reshape(-1, 3)
    return np.array(vals).reshape(-1, 2)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    left_eye = landmarks[[33, 133, 160, 159, 158, 144, 145, 153]].mean(axis=0)
    right_eye = landmarks[[362, 263, 387, 386, 385, 373, 374, 380]].mean(axis=0)
    eye_center = (left_eye + right_eye) / 2
    eye_dist = np.linalg.norm(left_eye[:2] - right_eye[:2]) + 1e-6
    normalized = (landmarks - eye_center) / eye_dist
    return normalized


def compute_distances(landmarks: np.ndarray, pairs: list) -> np.ndarray:
    distances = []
    for i, j in pairs:
        if i < len(landmarks) and j < len(landmarks):
            d = np.linalg.norm(landmarks[i] - landmarks[j])
            distances.append(d)
    return np.array(distances)


def compute_angles(landmarks: np.ndarray) -> np.ndarray:
    angles = []
    triplets = [
        (33, 168, 362),   # eye-bridge angle
        (70, 33, 61),     # left face angle
        (300, 362, 291),  # right face angle
        (61, 0, 291),     # mouth angle
        (33, 1, 362),     # nose-eyes angle
        (70, 168, 300),   # eyebrow angle
    ]
    for a, b, c in triplets:
        if max(a, b, c) < len(landmarks):
            v1 = landmarks[a][:2] - landmarks[b][:2]
            v2 = landmarks[c][:2] - landmarks[b][:2]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
    return np.array(angles)


def compute_ratios(landmarks: np.ndarray) -> np.ndarray:
    def dist(i, j):
        return np.linalg.norm(landmarks[i][:2] - landmarks[j][:2]) + 1e-6
    
    ratios = [
        dist(61, 291) / dist(0, 17),           # mouth width/height
        dist(159, 145) / dist(33, 133),        # left eye height/width
        dist(386, 374) / dist(362, 263),       # right eye height/width
        dist(70, 33) / dist(33, 362),          # eyebrow-eye / inter-eye
        dist(1, 0) / dist(1, 152),             # nose-lip / nose-chin
        dist(61, 291) / dist(33, 362),         # mouth width / inter-eye
        dist(70, 105) / dist(33, 362),         # eyebrow length / inter-eye
    ]
    return np.array(ratios)


def compute_region_stats(landmarks: np.ndarray) -> np.ndarray:
    stats = []
    for name, indices in LANDMARK_GROUPS.items():
        valid_idx = [i for i in indices if i < len(landmarks)]
        if len(valid_idx) < 2:
            stats.extend([0, 0, 0])
            continue
        region = landmarks[valid_idx]
        centroid = region.mean(axis=0)
        spread = np.std(region, axis=0).mean()
        aspect = (region[:, 0].max() - region[:, 0].min()) / (region[:, 1].max() - region[:, 1].min() + 1e-6)
        stats.extend([spread, aspect, np.linalg.norm(centroid[:2])])
    return np.array(stats)


def extract_features(landmark_str: str) -> np.ndarray:
    landmarks = parse_landmarks(landmark_str)
    normalized = normalize_landmarks(landmarks)
    
    key_coords = normalized[KEY_LANDMARKS].flatten()
    distances = compute_distances(normalized, DISTANCE_PAIRS)
    angles = compute_angles(normalized)
    ratios = compute_ratios(landmarks)
    region_stats = compute_region_stats(normalized)
    
    features = np.concatenate([key_coords, distances, angles, ratios, region_stats])
    return features.astype(np.float32)


class GeometricLandmarkDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, augment: bool = False):
        self.label_to_idx = label_to_idx
        self.augment = augment
        
        features, targets = [], []
        for _, row in df.iterrows():
            feat = extract_features(row['landmark_value'])
            if not np.isnan(feat).any():
                features.append(feat)
                targets.append(label_to_idx[row['emotion_label']])
        
        self.features = torch.tensor(np.stack(features), dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
        self.mean = self.features.mean(dim=0, keepdim=True)
        self.std = self.features.std(dim=0, keepdim=True) + 1e-6
        self.features = (self.features - self.mean) / self.std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feat = self.features[idx]
        if self.augment and self.training:
            feat = feat + torch.randn_like(feat) * 0.02
        return feat, self.targets[idx]
    
    @property
    def training(self):
        return self.augment


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(x + self.block(x))


class LandmarkClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [256, 256, 256, 128]):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.GELU(), nn.Dropout(0.3)
        )
        layers = []
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i+1]:
                layers.append(ResidualBlock(hidden_dims[i]))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]), nn.GELU(), nn.Dropout(0.2)
                ))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.layers(self.input_proj(x)))


def get_class_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).float()
    weights = 1.0 / (counts + 1)
    return weights / weights.sum() * num_classes


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    df = pd.read_csv(args.csv)
    if args.resolution:
        df = df[df["resolution_level"] == args.resolution]
    
    labels = sorted(df["emotion_label"].unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    dataset = GeometricLandmarkDataset(df, label_to_idx, augment=True)
    n = len(dataset)
    n_val, n_test = int(n * 0.1), int(n * 0.1)
    n_train = n - n_val - n_test
    
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                                  generator=torch.Generator().manual_seed(args.seed))
    
    train_targets = dataset.targets[train_set.indices]
    class_weights = get_class_weights(train_targets, len(labels)).to(device)
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    input_dim = dataset.features.shape[1]
    model = LandmarkClassifier(input_dim, len(labels), args.hidden_dims).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                                      epochs=args.epochs, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    best_val_acc, best_state, patience = 0, None, 0
    
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        
        if args.verbose:
            print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'input_dim': input_dim,
        'num_classes': len(labels),
        'label_to_idx': label_to_idx,
        'mean': dataset.mean,
        'std': dataset.std,
    }, save_dir / "model.pt")
    
    print(f"Features: {input_dim}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("rafdb_mediapipe_landmarks.csv"))
    parser.add_argument("--save_dir", type=Path, default=Path("geometric_models"))
    parser.add_argument("--resolution", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 256, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--early_stop", type=int, default=25)
    train(parser.parse_args())