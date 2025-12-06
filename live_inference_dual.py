import cv2
import torch
import numpy as np
import time
from collections import deque
from pathlib import Path
from model import HybridRNN
from landmark_features import LandmarkClassifier, extract_features, KEY_LANDMARKS
from mediapipe_landmarks import MediapipeLandmarkExtractor

class RealTimeEmotionDetector:
    def __init__(self, model_path, device='cuda', seq_len=10):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        print(f"Loading model on {self.device}...")

        # 1. Load Checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        self.idx_to_label = {v: k for k, v in ckpt['label_to_idx'].items()}
        self.mean = ckpt['mean'].to(self.device)
        self.std = ckpt['std'].to(self.device)
        
        # 2. Initialize Model Architecture
        old_model = LandmarkClassifier(
            input_dim=ckpt['input_dim'],
            num_classes=len(self.idx_to_label),
            hidden_dims=[256, 256, 256, 128]
        )
        self.model = HybridRNN(old_model, hidden_rnn_size=128, num_classes=len(self.idx_to_label))
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 3. Initialize Extractor
        self.extractor = MediapipeLandmarkExtractor()
        
        # 4. Feature Buffer (FIFO Queue)
        self.feature_buffer = deque(maxlen=seq_len)

    def process_frame(self, frame):
        """
        Extract landmarks -> Update buffer -> Predict (if buffer full)
        Returns: emotion (str), confidence (float), landmarks (np.array)
        """
        # A. Extract Landmarks using MediaPipe
        # extract_from_image returns normalized coordinates (0.0 - 1.0)
        landmarks = self.extractor.extract_from_image(frame)

        if landmarks is None:
            # Return None for landmarks if face not detected
            return None, 0.0, None

        # B. Prepare Features for Model
        # Convert to string format to reuse existing feature extraction logic
        landmark_str = ','.join([f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in landmarks])
        feat = extract_features(landmark_str) # Shape: (125,)

        # C. Update Buffer
        self.feature_buffer.append(feat)

        # D. Check Buffer Status
        if len(self.feature_buffer) < self.seq_len:
            return "Buffering...", 0.0, landmarks

        # E. Prepare Tensor Batch
        seq_features = np.array(self.feature_buffer)
        seq_tensor = torch.tensor(seq_features, dtype=torch.float32).to(self.device)
        
        # Apply Normalization (using training statistics)
        norm_features = (seq_tensor - self.mean) / self.std
        
        # Add Batch Dimension: [1, 10, 125]
        input_batch = norm_features.unsqueeze(0)

        # F. Run Inference
        with torch.no_grad():
            logits = self.model(input_batch)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred_idx].item()
            
        emotion = self.idx_to_label[pred_idx]

        # G. Confidence Thresholding
        if conf < 0.6:
            emotion = 'neutral'
            
        return emotion, conf, landmarks

    def close(self):
        self.extractor.close()

def main():
    # Configuration
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
    
    # Pre-calculate key landmark indices for O(1) lookup
    key_indices = set(KEY_LANDMARKS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        clean_frame = frame.copy()
        
        h, w, _ = frame.shape

        start_time = time.time()
        emotion, conf, landmarks = detector.process_frame(frame)
        
        if landmarks is not None:
            for i, lm in enumerate(landmarks):
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                
                if i in key_indices:
                    # Key landmarks: Red color, slightly larger
                    # BGR Color: (0, 0, 255) -> Red
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                else:
                    # Other landmarks: Green color, small dot
                    # BGR Color: (0, 255, 0) -> Green
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        #fps = 1.0 / (time.time() - start_time + 1e-6)
        fps = 3

        if emotion is None:
            status_text = "No Face Detected"
            color = (0, 0, 255)
        elif emotion == "Buffering...":
            status_text = f"Buffering {len(detector.feature_buffer)}/10"
            color = (255, 255, 0)
        else:
            current_emotion = emotion
            current_conf = conf
            
            # Change color based on confidence or forced neutral
            if current_conf < 0.6 and emotion == 'neutral':
                 status_text = f"NEUTRAL (Low Conf: {current_conf:.1%})"
                 color = (150, 150, 150) # Grey for uncertain fallback
            else:
                 status_text = f"{current_emotion.upper()} ({current_conf:.1%})"
                 color = (0, 255, 0) if current_conf > 0.8 else (0, 255, 255)
        
        # Top black bar
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # Status text
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow('Live LSTM Emotion Detector', frame)
        cv2.imshow('Raw Camera Feed', clean_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()