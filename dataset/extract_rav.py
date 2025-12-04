import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Emotion mapping from RAVDESS filename spec:
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad,
# 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def unzip_video_archives(zip_root: Path, extract_root: Path):
    """
    Recursively find all Video_*Actor*.zip under zip_root and unzip each
    into a subfolder of extract_root named after the zip (without extension).

    Example:
        1188976/Video_Song_Actor_01.zip
            -> extract_root/Video_Song_Actor_01/...
    """
    extract_root.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(zip_root.rglob("Video_*Actor*.zip"))
    if not zip_files:
        print(f"No Video_*Actor*.zip files found under {zip_root}")
        return

    for zpath in tqdm(zip_files, desc="Unzipping actor video archives"):
        dest_dir = extract_root / zpath.stem
        # Skip if already extracted and non-empty
        if dest_dir.exists() and any(dest_dir.iterdir()):
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dest_dir)


def sample_video_frames(video_path: Path, out_root: Path, num_frames: int = 10):
    """
    Sample num_frames evenly spaced frames from video_path and save them
    into emotion-specific folders inside out_root.
    """
    stem = video_path.stem  # e.g. "03-01-05-01-01-01-01"
    parts = stem.split("-")
    if len(parts) < 3:
        print(f"Skipping {video_path.name}: unexpected filename format")
        return 0

    emotion_code = parts[2]
    emotion = EMOTION_MAP.get(emotion_code)
    if emotion is None:
        print(f"Skipping {video_path.name}: unknown emotion code {emotion_code}")
        return 0

    emotion_dir = out_root / emotion
    emotion_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: cannot open video {video_path}")
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        print(f"Warning: no frames found in {video_path}")
        return 0

    num_to_sample = min(num_frames, frame_count)
    indices = np.linspace(0, frame_count - 1, num=num_to_sample, dtype=int)

    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        out_name = f"{stem}_frame{i:02d}_{emotion}.png"
        out_path = emotion_dir / out_name
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def process_all_videos(extract_root: Path, out_root: Path, num_frames: int = 10):
    """
    Find all videos under extract_root (recursively) and sample frames
    into out_root/emotion_name/.
    """
    video_files = [
        p for p in extract_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    if not video_files:
        print(f"No video files found under {extract_root}")
        return

    total_saved = 0
    for vpath in tqdm(video_files, desc="Processing videos"):
        total_saved += sample_video_frames(vpath, out_root, num_frames=num_frames)

    print(f"Done. Saved {total_saved} images into {out_root}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unzip RAVDESS archive, unzip per-actor video zips, "
            "sample frames, and organize them into emotion folders."
        )
    )
    parser.add_argument(
        "--zip_root",
        type=str,
        default="1188976.zip",
        help=(
            "Path to the TOP-LEVEL RAVDESS zip (e.g. 1188976.zip), "
            "or to the already-unzipped folder (e.g. 1188976/)."
        ),
    )
    parser.add_argument(
        "--extract_root",
        type=str,
        default="./ravdess_videos",
        help="Folder where Video_*Actor* subfolders with videos will be created.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./ravdess_emotion_frames",
        help="Folder where emotion subfolders with extracted images will be created.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of frames to sample from each video.",
    )
    parser.add_argument(
        "--skip_unzip",
        action="store_true",
        help=(
            "Skip ALL unzipping (assume videos already extracted under extract_root)."
        ),
    )

    args = parser.parse_args()

    zip_root_arg = Path(args.zip_root)
    extract_root = Path(args.extract_root)
    out_root = Path(args.out_root)

    # --- Step 1: top-level unzip (1188976.zip -> 1188976/) ---
    if not args.skip_unzip:
        if zip_root_arg.is_file() and zip_root_arg.suffix == ".zip":
            top_dir = zip_root_arg.with_suffix("")  # e.g. 1188976
            if not top_dir.exists() or not any(top_dir.iterdir()):
                print(f"Unzipping top-level archive {zip_root_arg} -> {top_dir}")
                top_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_root_arg, "r") as zf:
                    zf.extractall(top_dir)
            zip_root = top_dir
        elif zip_root_arg.is_dir():
            # Already unzipped, use this folder directly
            zip_root = zip_root_arg
        else:
            print(f"zip_root path {zip_root_arg} is neither a zip file nor a directory.")
            return

        # --- Step 2: unzip all Video_*Actor*.zip inside that folder ---
        unzip_video_archives(zip_root, extract_root)

    # --- Step 3: process videos into emotion folders ---
    process_all_videos(extract_root, out_root, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
