import cv2
import torch
import numpy as np
import time
from collections import deque
from pathlib import Path
from model import HybridRNN
from landmark_features import LandmarkClassifier, extract_features
from mediapipe_landmarks import MediapipeLandmarkExtractor

class RealTimeEmotionDetector:
    def __init__(self, model_path, device='cuda', seq_len=10):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        print(f"Loading model on {self.device}...")

        ckpt = torch.load(model_path, map_location=self.device)
        self.idx_to_label = {v: k for k, v in ckpt['label_to_idx'].items()}
        self.mean = ckpt['mean'].to(self.device)
        self.std = ckpt['std'].to(self.device)
        
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
        self.feature_buffer = deque(maxlen=seq_len)

    def process_frame(self, frame):
        landmarks = self.extractor.extract_from_image(frame)

        if landmarks is None:
            return None, 0.0

        landmark_str = ','.join([f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in landmarks])
        feat = extract_features(landmark_str) # numpy array (125,)

        self.feature_buffer.append(feat)

        if len(self.feature_buffer) < self.seq_len:
            return "Buffering...", 0.0

        seq_features = np.array(self.feature_buffer)
        seq_tensor = torch.tensor(seq_features, dtype=torch.float32).to(self.device)
        
        norm_features = (seq_tensor - self.mean) / self.std
        
        input_batch = norm_features.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_batch)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred_idx].item()
            
        emotion = self.idx_to_label[pred_idx]
        return emotion, conf

    def close(self):
        self.extractor.close()

def main():
    MODEL_PATH = 'temporal_models/lstm_model_best.pt'
    CAMERA_INDEX = 0
    
    if not Path(MODEL_PATH).exists():
        print("Error: Model not found.")
        return

    detector = RealTimeEmotionDetector(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    print("Starting Live Emotion Recognition... Press 'q' to quit.")
    
    current_emotion = "Waiting..."
    current_conf = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        start_time = time.time()
        emotion, conf = detector.process_frame(frame)
        fps = 1.0 / (time.time() - start_time + 1e-6)

        if emotion is None:
            status_text = "No Face Detected"
        elif emotion == "Buffering...":
            status_text = f"Buffering {len(detector.feature_buffer)}/10"
        else:
            current_emotion = emotion
            current_conf = conf
            status_text = f"{current_emotion.upper()} ({current_conf:.1%})"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        
        color = (0, 255, 0) if current_conf > 0.7 else (0, 255, 255)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Live LSTM Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()