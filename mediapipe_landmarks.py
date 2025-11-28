import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

class MediapipeLandmarkExtractor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
    
    def extract(self, image_path: Path) -> np.ndarray | None:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
        return landmarks
    
    def extract_from_image(self, img: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])

    def close(self):
        self.face_mesh.close()


class RAFDBMediapipeProcessor:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.extractor = MediapipeLandmarkExtractor()
        self.blur_thresholds = {'not_blurry': 1420.2, 'medium_blurry': 552.9}
    
    def calculate_blurriness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def classify_blur(self, score: float) -> str:
        if score >= self.blur_thresholds['not_blurry']:
            return 'not_blurry'
        elif score >= self.blur_thresholds['medium_blurry']:
            return 'medium_blurry'
        return 'blurry'
    
    def process_dataset(self) -> pd.DataFrame:
        results = []
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
            for emotion_folder in tqdm(list(split_path.iterdir()), desc=f"{split}"):
                if not emotion_folder.is_dir():
                    continue
                emotion = emotion_folder.name
                for img_path in list(emotion_folder.glob('*.jpg')) + list(emotion_folder.glob('*.png')):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    landmarks = self.extractor.extract(img_path)
                    if landmarks is None:
                        continue
                    blur_score = self.calculate_blurriness(img)
                    landmark_str = ','.join([f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in landmarks])
                    results.append({
                        'picture_id': img_path.name,
                        'emotion_label': emotion,
                        'resolution_level': self.classify_blur(blur_score),
                        'landmark_value': landmark_str,
                        'split': split
                    })
        self.extractor.close()
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} samples to {output_path}")
        print(f"Emotions: {df['emotion_label'].value_counts().to_dict()}")
        print(f"Resolution: {df['resolution_level'].value_counts().to_dict()}")


def main():
    DATA_PATH = '/Users/test/DL/rafdb'
    OUTPUT_CSV = '/Users/test/subject-aware-temporal-FER/rafdb_mediapipe_landmarks.csv'
    processor = RAFDBMediapipeProcessor(DATA_PATH)
    df = processor.process_dataset()
    if len(df) > 0:
        processor.save_results(df, OUTPUT_CSV)

if __name__ == "__main__":
    main()