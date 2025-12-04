import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
import os
import torchvision.models as models
import torchvision.transforms as transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(2): self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7): self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12): self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21): self.slice4.add_module(str(x), vgg[x])
            
        self.slice1.eval().to(device)
        self.slice2.eval().to(device)
        self.slice3.eval().to(device)
        self.slice4.eval().to(device)
        
        for param in self.parameters():
            param.requires_grad = False

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        # x, y range [-1, 1] -> [0, 1]
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5
        x = self.transform(x)
        y = self.transform(y)

        h_x1 = self.slice1(x); h_y1 = self.slice1(y)
        h_x2 = self.slice2(h_x1); h_y2 = self.slice2(h_y1)
        h_x3 = self.slice3(h_x2); h_y3 = self.slice3(h_y2)
        h_x4 = self.slice4(h_x3); h_y4 = self.slice4(h_y3)

        loss = F.l1_loss(h_x1, h_y1) * 1.0 + \
               F.l1_loss(h_x2, h_y2) * 1.5 + \
               F.l1_loss(h_x3, h_y3) * 2.0 + \
               F.l1_loss(h_x4, h_y4) * 2.5
        return loss
    
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, ngf=64, n_res=6):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True)
        )
        self.res_blocks = nn.Sequential(*[ResBlock(ngf*4) for _ in range(n_res)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, 7), nn.Tanh()
        )
    def forward(self, x, noise_map):
        combined = torch.cat([x, noise_map], dim=1)
        return self.decoder(self.res_blocks(self.encoder(combined)))

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

class FaceIDAugmentor:
    def __init__(self, device='cuda', img_size=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        self.perceptual_net = PerceptualLoss(self.device)
        
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def get_noise_map(self, x, strength=1.0):
        b, _, h, w = x.shape
        return torch.randn(b, 1, h, w).to(self.device) * strength

    def train_step(self, real_imgs: torch.Tensor):

        noise1 = self.get_noise_map(real_imgs, strength=1.0)
        fake1 = self.G(real_imgs, noise1)
        
        self.opt_D.zero_grad()
        loss_D_real = F.relu(1.0 - self.D(real_imgs)).mean()
        loss_D_fake = F.relu(1.0 + self.D(fake1.detach())).mean()
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.opt_D.step()
        
        self.opt_G.zero_grad()
        
        noise2 = self.get_noise_map(real_imgs, strength=1.0)
        fake2 = self.G(real_imgs, noise2)
        
        loss_adv = -self.D(fake1).mean()
        loss_perc = self.perceptual_net(fake1, real_imgs)
        loss_pixel = F.l1_loss(fake1, real_imgs) * 0.5
        loss_div = -torch.mean(torch.abs(fake1 - fake2)) * 2.0
        loss_G = loss_adv + (loss_perc * 5.0) + loss_pixel + loss_div
        
        loss_G.backward()
        self.opt_G.step()
        
        return {
            'D': loss_D.item(), 
            'G_adv': loss_adv.item(), 
            'G_perc': loss_perc.item(), 
            'G_div': loss_div.item()
        }

    def augment(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        self.G.eval()
        with torch.no_grad():
            x = self.preprocess(img)
            noise_map = self.get_noise_map(x, strength=strength)
            aug = self.G(x, noise_map)
        return self.postprocess(aug)

    def save(self, path: str):
        torch.save({'G': self.G.state_dict(), 'D': self.D.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])


class DatasetAugmenter:
    def __init__(self, data_path: str, augmentor: FaceIDAugmentor):
        self.data_path = Path(data_path)
        self.augmentor = augmentor

    def train_augmentor(self, epochs: int = 50, batch_size: int = 8):
        images = []
        train_path = self.data_path / 'train'
        
        print("Loading dataset for training...")
        count = 0
        limit = 2000 
        
        image_paths = []
        if train_path.exists():
            for emotion in train_path.iterdir():
                if emotion.is_dir():
                    image_paths.extend(list(emotion.glob('*.jpg')))
        
        np.random.shuffle(image_paths)
        image_paths = image_paths[:limit]
        
        for p in tqdm(image_paths, desc="Loading RAM"):
            img = cv2.imread(str(p))
            if img is not None:
                images.append(self.augmentor.preprocess(img).cpu())
        
        if not images:
            print("No images found.")
            return

        dataset = torch.cat(images, dim=0)
        print(f"Training on {len(dataset)} images for {epochs} epochs...")
        
        for epoch in range(epochs):
            perm = torch.randperm(len(dataset))
            pbar = tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
            for i in pbar:
                batch = dataset[perm[i:i+batch_size]].to(self.augmentor.device)
                losses = self.augmentor.train_step(batch)
                pbar.set_postfix(D=f"{losses['D']:.2f}", Div=f"{losses['G_div']:.2f}", Perc=f"{losses['G_perc']:.2f}")

    def augment_dataset(self, output_path: Path, strengths: list):
        output_path.mkdir(parents=True, exist_ok=True)
        for split in ['train']:
            split_path = self.data_path / split
            if not split_path.exists(): continue
            
            print(f"Augmenting {split} set...")
            for emotion_folder in tqdm(list(split_path.iterdir()), desc=split):
                if not emotion_folder.is_dir(): continue
                out_emotion = output_path / split / emotion_folder.name
                out_emotion.mkdir(parents=True, exist_ok=True)
                
                for img_path in emotion_folder.glob('*.jpg'):
                    img = cv2.imread(str(img_path))
                    if img is None: continue
                    
                    cv2.imwrite(str(out_emotion / img_path.name), img)
                    
                    for i, s in enumerate(strengths):
                        aug_img = self.augmentor.augment(img, strength=s)
                        aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                        cv2.imwrite(str(out_emotion / aug_name), aug_img)

def main():
    DATA_PATH = 'C:\\794project_dataset\\RAF_DB'
    OUTPUT_PATH = Path('C:\\794project_dataset\\RAF_DB_augmented_v2')
    MODEL_PATH = '../faceid_models/faceid_gan_diversity.pt'

    if not OUTPUT_PATH.exists(): OUTPUT_PATH.mkdir(parents=True)
    if not Path(MODEL_PATH).parent.exists(): Path(MODEL_PATH).parent.mkdir(parents=True)
    
    augmentor = FaceIDAugmentor(device='cuda')
    dataset_aug = DatasetAugmenter(DATA_PATH, augmentor)
    
    print("Training FaceID-GAN with Diversity Loss...")
    dataset_aug.train_augmentor(epochs=30, batch_size=8)
    augmentor.save(MODEL_PATH)
    
    print("Augmenting dataset...")
    ten_strengths = np.linspace(0.5, 1.5, 10).tolist()
    dataset_aug.augment_dataset(OUTPUT_PATH, strengths=ten_strengths)
    print("Done!")

if __name__ == "__main__":
    main()