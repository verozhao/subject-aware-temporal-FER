import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import re
from pathlib import Path
from tqdm import tqdm

from landmark_features import LandmarkClassifier, extract_features
from mediapipe_landmarks import MediapipeLandmarkExtractor
from model import HybridRNN


class VideoInferencePipeline:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading LSTM model on {self.device}...")
        
        ckpt = torch.load(model_path, map_location=self.device)
        
        self.idx_to_label = {v: k for k, v in ckpt['label_to_idx'].items()}
        self.mean = ckpt['mean'].to(self.device)
        self.std = ckpt['std'].to(self.device)
        self.seq_len = ckpt.get('seq_len', 10)
        
        old_model = LandmarkClassifier(
            input_dim=ckpt['input_dim'],
            num_classes=len(self.idx_to_label), 
            hidden_dims=[256, 256, 256, 128]
        )
        
        self.model = HybridRNN(old_model, hidden_rnn_size=128, num_classes=len(self.idx_to_label))
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.extractor = MediapipeLandmarkExtractor()

    def _extract_sequence_features(self, image_paths):
        feature_list = []
        valid_indices = []
        
        for i, img_path in enumerate(image_paths):
            landmarks = self.extractor.extract(img_path)
            
            if landmarks is None:
                print(f"Warning: No face detected in {img_path.name}, skipping frame.")
                feature_list.append(np.zeros(125))
            else:
                landmark_str = ','.join([f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in landmarks])
                
                feat = extract_features(landmark_str)
                feature_list.append(feat)
                valid_indices.append(i)
                
        return np.array(feature_list)

    def process_folder(self, folder_path: str):
        path = Path(folder_path)
        if not path.exists():
            print(f"Error: Folder {folder_path} does not exist.")
            return

        image_files = sorted(
            list(path.glob('*.png')) + list(path.glob('*.jpg')),
            key=lambda p: int(re.search(r'(\d+)(?=\.\w+$)', p.name).group()) if re.search(r'(\d+)(?=\.\w+$)', p.name) else 0
        )
        
        num_frames = len(image_files)
        print(f"\nFound {num_frames} frames in {path.name}")
        
        if num_frames < self.seq_len:
            print(f"Skipping: Not enough frames ({num_frames} < {self.seq_len})")
            return
            
        all_features = self._extract_sequence_features(image_files) # Shape: [N, 125]
        
        all_features_tensor = torch.tensor(all_features, dtype=torch.float32).to(self.device)
        
        norm_features = (all_features_tensor - self.mean) / self.std
        
        results = []
        
        loop_range = num_frames - self.seq_len + 1
        
        print(f"Running LSTM inference on {loop_range} windows...\n")
        print(f"{'Window Range':<20} | {'Prediction':<10} | {'Confidence':<10}")
        print("-" * 50)
        
        with torch.no_grad():
            for i in range(loop_range):
                window = norm_features[i : i + self.seq_len] # Shape: [10, 125]
                
                input_batch = window.unsqueeze(0)
                logits = self.model(input_batch)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred_idx].item()
                
                emotion = self.idx_to_label[pred_idx]
                
                start_name = image_files[i].name
                end_name = image_files[i + self.seq_len - 1].name
                
                print(f"{start_name[:15]}..{end_name[-10:]:<5} | {emotion.upper():<10} | {conf:.2%}")
                
                results.append((start_name, end_name, emotion, conf))
                
        return results

    def close(self):
        self.extractor.close()

def main():
    MODEL_PATH = 'temporal_models/lstm_model_best.pt'
    TARGET_FOLDER = 'C:\\Documents\\CMU\\Courses\\Fall2025\\18-794\\Project\\subject-aware-temporal-FER\\video_0'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=TARGET_FOLDER, help="Folder containing video frames")
    parser.add_argument('--model', type=str, default=MODEL_PATH, help="Path to .pt model file")
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Model file not found: {args.model}")
        return

    pipeline = VideoInferencePipeline(args.model)
    
    try:
        pipeline.process_folder(args.in_dir)
    finally:
        pipeline.close()

if __name__ == "__main__":
    main()