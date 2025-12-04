import cv2
import sys
import os
import argparse
import numpy as np
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from mediapipe_landmarks import MediapipeLandmarkExtractor
from landmark_features import KEY_LANDMARKS

def draw_landmarks(image_path, output_dir):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Input image not found at {img_path}")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    extractor = MediapipeLandmarkExtractor()

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
    
    h, w, _ = img.shape
    print(f"Processing: {img_path.name} ({w}x{h})")

    full_landmarks = extractor.extract(img_path)

    if full_landmarks is None:
        print("  -> No face detected.")
        return

    print(f"  -> Detected face. Highlighting {len(KEY_LANDMARKS)} key landmarks...")

    for idx in KEY_LANDMARKS:
        if idx < len(full_landmarks):
            lm = full_landmarks[idx]
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            
            cv2.circle(img, (cx, cy), 1, (0, 0, 255), -1)

    output_file = out_path / f"vis_{img_path.name}"
    cv2.imwrite(str(output_file), img)
    print(f"  -> Saved to: {output_file}")

    extractor.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize specific FER landmarks on an image.")
    
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the input image file')
    parser.add_argument('--out_dir', type=str, default='../samples/landmarks_samples', 
                        help='Directory to save the annotated image')

    args = parser.parse_args()
    
    draw_landmarks(args.image, args.out_dir)

if __name__ == "__main__":
    main()