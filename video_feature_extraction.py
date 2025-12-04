import cv2
import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from mediapipe_landmarks import MediapipeLandmarkExtractor

random.seed(42)

class VideoDatasetProcessor:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.extractor = MediapipeLandmarkExtractor()
        
    def get_video_id(self, filename: str) -> str:
        """
        Complete Format: '01-01-05-01-01-01-02_frame00_angry.png'
        Video Format: '01-01-05-01-01-01-02'
        """
        if '_frame' in filename:
            return filename.split('_frame')[0]
        else:
            return filename.split('.')[0]

    def process_dataset(self) -> pd.DataFrame:
        if not self.data_path.exists():
            print(f"Error: Data path {self.data_path} does not exist.")
            return pd.DataFrame()

        print("Scanning files and grouping by video ID...")
        video_dict = {} # {video_id: [file_path1, file_path2, ...]}
        
        emotion_folders = [f for f in self.data_path.iterdir() if f.is_dir()]
        
        for emotion_folder in emotion_folders:
            image_files = list(emotion_folder.glob('*.jpg')) + list(emotion_folder.glob('*.png'))
            for img_path in image_files:
                vid_id = self.get_video_id(img_path.name)
                
                if vid_id not in video_dict:
                    video_dict[vid_id] = []
                
                video_dict[vid_id].append({
                    'path': img_path,
                    'emotion': emotion_folder.name
                })
        
        all_video_ids = list(video_dict.keys())
        print(f"Found {len(all_video_ids)} unique videos.")
        
        print("Splitting videos into Train/Val/Test...")
        random.shuffle(all_video_ids)
        
        n_videos = len(all_video_ids)
        n_train = int(n_videos * 0.7)
        n_val = int(n_videos * 0.2)
        
        train_ids = set(all_video_ids[:n_train])
        val_ids = set(all_video_ids[n_train : n_train + n_val])
        test_ids = set(all_video_ids[n_train + n_val:])
        
        print(f"Split counts -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

        results = []

        def get_split_label(vid_id):
            if vid_id in train_ids: return 'train'
            if vid_id in val_ids: return 'val'
            return 'test'

        for vid_id in tqdm(all_video_ids, desc="Extracting Landmarks"):
            split_label = get_split_label(vid_id)
            frames = video_dict[vid_id]
            
            for frame_info in frames:
                img_path = frame_info['path']
                emotion = frame_info['emotion']
                
                landmarks = self.extractor.extract(img_path)
                
                if landmarks is None:
                    continue
                landmark_str = ','.join([f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in landmarks])
                
                results.append({
                    'picture_id': img_path.name,
                    'emotion_label': emotion,
                    'resolution_level': 'none',
                    'landmark_value': landmark_str,
                    'split': split_label
                })

        self.extractor.close()
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved processed data to {output_path}")
        print(f"Total samples: {len(df)}")
        print("Split distribution:")
        print(df['split'].value_counts())

def main():
    DATA_PATH = 'C:\\Users\\admin\\Downloads\\ravdess_emotion_frames'
    OUTPUT_CSV = 'C:\\Documents\\CMU\\Courses\\Fall2025\\18-794\\Project\\subject-aware-temporal-FER\\landmark_features\\ravdess_landmarks_split.csv'

    if not Path(DATA_PATH).exists():
        print(f"Data path {DATA_PATH} does not exist.")
        return
    
    if not Path(OUTPUT_CSV).parent.exists():
        print(f"Output directory {Path(OUTPUT_CSV).parent} does not exist.")
        return
    
    processor = VideoDatasetProcessor(DATA_PATH)
    df = processor.process_dataset()
    
    if len(df) > 0:
        processor.save_results(df, OUTPUT_CSV)
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()