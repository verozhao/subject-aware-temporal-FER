import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, ngf=64, n_res=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, ngf, 7, 1, 3), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True)
        )
        self.res_blocks = nn.Sequential(*[ResBlock(ngf*4) for _ in range(n_res)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, 3, 7, 1, 3), nn.Tanh()
        )
    def forward(self, x):
        return self.decoder(self.res_blocks(self.encoder(x)))

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, 1, 1), nn.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.net(x)

class IdentityEncoder(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class FaceIDAugmentor:
    def __init__(self, device='cuda', img_size=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        self.id_encoder = IdentityEncoder().to(self.device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def train_step(self, real_imgs: torch.Tensor):
        noise = torch.randn_like(real_imgs) * 0.1
        noisy_input = real_imgs + noise
        
        fake_imgs = self.G(noisy_input)
        self.opt_D.zero_grad()
        loss_D = (F.relu(1 - self.D(real_imgs)).mean() + F.relu(1 + self.D(fake_imgs.detach())).mean())
        loss_D.backward()
        self.opt_D.step()
        
        self.opt_G.zero_grad()
        fake_imgs = self.G(noisy_input)
        loss_adv = -self.D(fake_imgs).mean()
        loss_rec = F.l1_loss(fake_imgs, real_imgs) * 10
        id_real, id_fake = self.id_encoder(real_imgs), self.id_encoder(fake_imgs)
        loss_id = (1 - F.cosine_similarity(id_real, id_fake)).mean() * 5
        loss_G = loss_adv + loss_rec + loss_id
        loss_G.backward()
        self.opt_G.step()
        return {'D': loss_D.item(), 'G': loss_G.item(), 'id': loss_id.item()}

    def augment(self, img: np.ndarray, strength: float = 0.15) -> np.ndarray:
        self.G.eval()
        with torch.no_grad():
            x = self.preprocess(img)
            noise = torch.randn_like(x) * strength
            aug = self.G(x + noise)
        return self.postprocess(aug)

    def save(self, path: str):
        torch.save({'G': self.G.state_dict(), 'D': self.D.state_dict(), 'id': self.id_encoder.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.id_encoder.load_state_dict(ckpt['id'])


class DatasetAugmenter:
    def __init__(self, data_path: str, augmentor: FaceIDAugmentor, aug_factor: int = 2):
        self.data_path = Path(data_path)
        self.augmentor = augmentor
        self.aug_factor = aug_factor

    def train_augmentor(self, epochs: int = 50, batch_size: int = 16):
        images = []
        for split in ['train']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
            for emotion_folder in split_path.iterdir():
                if emotion_folder.is_dir():
                    for img_path in list(emotion_folder.glob('*.jpg'))[:500]:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            images.append(self.augmentor.preprocess(img))
        
        dataset = torch.cat(images, dim=0)
        for epoch in range(epochs):
            perm = torch.randperm(len(dataset))
            losses = []
            for i in range(0, len(dataset), batch_size):
                batch = dataset[perm[i:i+batch_size]].to(self.augmentor.device)
                losses.append(self.augmentor.train_step(batch))
            print(f"Epoch {epoch+1}: D={np.mean([l['D'] for l in losses]):.4f} G={np.mean([l['G'] for l in losses]):.4f}")

    def augment_dataset(self, output_path: Path, strengths: list = [0.1, 0.2]):
        output_path.mkdir(parents=True, exist_ok=True)
        for split in ['train']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
            for emotion_folder in tqdm(list(split_path.iterdir()), desc=split):
                if not emotion_folder.is_dir():
                    continue
                out_emotion = output_path / split / emotion_folder.name
                out_emotion.mkdir(parents=True, exist_ok=True)
                for img_path in emotion_folder.glob('*.jpg'):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    cv2.imwrite(str(out_emotion / img_path.name), img)
                    for i, s in enumerate(strengths):
                        aug_img = self.augmentor.augment(img, strength=s)
                        aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                        cv2.imwrite(str(out_emotion / aug_name), aug_img)


def main():
    DATA_PATH = '/Users/test/DL/rafdb'
    OUTPUT_PATH = Path('/Users/test/DL/rafdb_augmented')
    MODEL_PATH = '/Users/test/DL/faceid_gan.pt'
    
    augmentor = FaceIDAugmentor(device='cuda')
    dataset_aug = DatasetAugmenter(DATA_PATH, augmentor)
    
    print("Training FaceID-GAN...")
    dataset_aug.train_augmentor(epochs=30)
    augmentor.save(MODEL_PATH)
    
    print("Augmenting dataset...")
    dataset_aug.augment_dataset(OUTPUT_PATH)
    print("Done!")

if __name__ == "__main__":
    main()