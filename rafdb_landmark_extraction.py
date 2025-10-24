import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm


class LandmarkExtractor:
    """Extract landmarks using face detection model"""
    def __init__(self, model_path: str, device='cuda', code_path='/Users/test/subject-aware-temporal-FER/model_code'):
        original_dir = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            sys.path.insert(0, code_path)
            os.chdir(code_path)
            
            from data import cfg
            from loss_and_anchor import anchor
            from detector.mydetector import mydetector
            from utils.nms import nms
            from utils.box_utils import decode, decode_landm
            from utils.misc import load_model
            
            self.cfg = cfg
            self.anchor = anchor
            self.mydetector = mydetector
            self.nms = nms
            self.decode = decode
            self.decode_landm = decode_landm
            self.load_model = load_model
            
        finally:
            os.chdir(original_dir)
            sys.path[:] = original_path
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        self.net = self.mydetector(cfg=self.cfg, phase='test')
        self.net = self.load_model(self.net, model_path, self.device == torch.device('cpu'))
        self.net.eval()
        self.net = self.net.to(self.device)
        
        torch.set_grad_enabled(False)
    
    def extract(self, image_path: Path):
        """Extract landmarks from image"""
        try:
            img_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_raw is None:
                return None
            
            img = np.float32(img_raw)
            
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)
            
            loc, conf, landms = self.net(img)
            priorbox = self.anchor(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward().to(self.device)
            prior_data = priors.data
            
            boxes = self.decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            
            landms = self.decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]]).to(self.device)
            landms = landms * scale1
            landms = landms.cpu().numpy()
            
            inds = np.where(scores > self.confidence_threshold)[0]
            if len(inds) == 0:
                return None
                
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = self.nms(dets, self.nms_threshold)
            
            if len(keep) == 0:
                return None
            
            best_idx = keep[0]
            landmarks = landms[best_idx].reshape(-1, 2)
            landmarks[:, 0] /= im_width
            landmarks[:, 1] /= im_height
            
            return landmarks
            
        except Exception as e:
            print(f"Error extracting landmarks from {image_path.name}: {e}")
            return None


class RAFDBProcessor:
    """Process RAF-DB dataset"""
    def __init__(self, data_path: str, model_path: str, code_path: str, min_landmarks: int = 5):
        self.data_path = Path(data_path)
        self.min_landmarks = min_landmarks
        self.landmark_extractor = LandmarkExtractor(model_path, code_path=code_path)
        self.blur_thresholds = {'not_blurry': 1420.2, 'medium_blurry': 552.9}
    
    def calculate_blurriness(self, image: np.ndarray) -> float:
        """Calculate Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def classify_blur(self, score: float) -> str:
        """Classify blur level"""
        if score >= self.blur_thresholds['not_blurry']:
            return 'not_blurry'
        elif score >= self.blur_thresholds['medium_blurry']:
            return 'medium_blurry'
        return 'blurry'
    
    def process_image(self, image_path: Path, emotion: str, split: str):
        """Process single image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            blur_score = self.calculate_blurriness(image)
            blur_level = self.classify_blur(blur_score)
            
            landmarks = self.landmark_extractor.extract(image_path)
            if landmarks is None or len(landmarks) < self.min_landmarks:
                return None
            
            landmark_str = ','.join([f"{x:.6f},{y:.6f}" for x, y in landmarks])
            
            return {
                'picture_id': image_path.name,
                'emotion_label': emotion,
                'resolution_level': blur_level,
                'landmark_value': landmark_str
            }
        
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            return None
    
    def process_dataset(self) -> pd.DataFrame:
        """Process entire dataset"""
        results = []
        
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                print(f"Warning: {split} not found")
                continue
            
            emotion_folders = [d for d in split_path.iterdir() if d.is_dir()]
            
            print(f"\nProcessing {split} split...")
            for emotion_folder in tqdm(emotion_folders, desc=f"{split}"):
                emotion = emotion_folder.name
                images = list(emotion_folder.glob('*.jpg')) + list(emotion_folder.glob('*.png'))
                
                for image_path in images:
                    result = self.process_image(image_path, emotion, split)
                    if result:
                        results.append(result)
        
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save results"""
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Complete! Total: {len(df)}")
        print(f"\nEmotion:\n{df['emotion_label'].value_counts()}")
        print(f"\nResolution:\n{df['resolution_level'].value_counts()}")
        print(f"\nSaved: {output_path}")
        print(f"{'='*60}")


def main():

    CODE_PATH = '/Users/test/subject-aware-temporal-FER/model_code'
    DATA_PATH = '/Users/test/DL/rafdb'
    MODEL_PATH = '/Users/test/DL/weights_cn20_2gpuResnet50_Final.pth'
    OUTPUT_CSV = '/Users/test/subject-aware-temporal-FER/rafdb_landmarks_processed.csv'
    MIN_LANDMARKS = 5
    
    print("RAF-DB Landmark Extraction")
    print("="*60)
    
    processor = RAFDBProcessor(DATA_PATH, MODEL_PATH, CODE_PATH, min_landmarks=MIN_LANDMARKS)
    df = processor.process_dataset()
    
    if len(df) > 0:
        processor.save_results(df, OUTPUT_CSV)
    else:
        print("No valid images processed!")


if __name__ == "__main__":
    main()