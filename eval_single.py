import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from landmark_features import extract_features
from model import LandmarkClassifier
from mediapipe_landmarks import MediapipeLandmarkExtractor


class InferencePipeline:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"Loading pipeline on {self.device}...")
        ckpt = torch.load(model_path, map_location=self.device)

        self.idx_to_label = {v: k for k, v in ckpt['label_to_idx'].items()}
        self.mean = ckpt['mean'].to(self.device)
        self.std = ckpt['std'].to(self.device)
        
        self.model = LandmarkClassifier(
            input_dim=ckpt['input_dim'],
            num_classes=ckpt['num_classes'],
            hidden_dims=[256, 256, 256, 128] 
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.extractor = MediapipeLandmarkExtractor()
        
    def predict(self, image_path: str):
        raw_landmarks = self.extractor.extract(Path(image_path))
        
        if raw_landmarks is None:
            print(f"Error: No face detected in {image_path}")
            return None

        landmark_str = ','.join([f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in raw_landmarks])
        features = extract_features(landmark_str) # Shape: (125,)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        input_tensor = features_tensor.unsqueeze(0) 
        
        input_tensor = (input_tensor - self.mean) / self.std
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        prediction = self.idx_to_label[pred_idx]
        
        return prediction, confidence

    def close(self):
        self.extractor.close()

def main():
    MODEL_PATH = 'C:\\Documents\\CMU\\Courses\\Fall2025\\18-794\\Project\\subject-aware-temporal-FER\\geometric_models\\model.pt'
    #TEST_IMAGE_PATH = 'C:\\794project_dataset\\RAF_DB\\test\\angry\\aug_101186.png'
    TEST_IMAGE_PATH = 'C:\\Users\\admin\\Downloads\\ravdess_emotion_frames\\angry\\01-01-05-01-01-01-02_frame04_angry.png'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--image', type=str, default=TEST_IMAGE_PATH)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found at {args.model}")
        return
    if not Path(args.image).exists():
        print(f"Image not found at {args.image}")
        return

    pipeline = InferencePipeline(args.model)
    
    try:
        print(f"\nProcessing image: {Path(args.image).name}")
        result = pipeline.predict(args.image)
        
        if result:
            emotion, conf = result
            print(f"Predicted Class: {emotion.upper()}")
            print(f"Confidence:      {conf:.2%}")
            
    finally:
        pipeline.close()

if __name__ == "__main__":
    main()