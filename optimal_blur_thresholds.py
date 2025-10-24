"""
To calculate optimal blur thresholds
"""
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def calculate_blurriness(image_path):
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return cv2.Laplacian(image, cv2.CV_64F).var()
    except:
        return None


def calculate_all_blur_scores(csv_path):
    df = pd.read_csv(csv_path)
    blur_scores = []
    data_path = Path('/Users/test/DL/rafdb')
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        picture_id = row['picture_id']
        emotion = row['emotion_label']
        
        for split in ['train', 'val', 'test']:
            for ext in ['.jpg', '.png']:
                base_name = picture_id.replace('.jpg', '').replace('.png', '')
                img_path = data_path / split / emotion / (base_name + ext)
                
                if img_path.exists():
                    blur_score = calculate_blurriness(img_path)
                    if blur_score is not None:
                        blur_scores.append(blur_score)
                    break
            else:
                continue
            break
    
    return np.array(blur_scores)


def main():
    csv_path = '/Users/test/subject-aware-temporal-FER/rafdb_landmarks_processed.csv'
    
    blur_scores = calculate_all_blur_scores(csv_path)
    
    sorted_scores = np.sort(blur_scores)
    n = len(sorted_scores)
    
    threshold_low = sorted_scores[n // 3]
    threshold_high = sorted_scores[2 * n // 3]

    print(f"Optimal low threshold: {threshold_low:.1f}")
    print(f"Optimal high threshold: {threshold_high:.1f}")
    
    df = pd.read_csv(csv_path)
    df['blur_score'] = blur_scores
    output_path = csv_path.replace('.csv', '_with_blur_scores.csv')
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()