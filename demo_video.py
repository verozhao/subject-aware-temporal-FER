import cv2
import os
import time
from collections import deque

# ====== Configs ======
CAPTURE_INTERVAL = 0.1      # seconds between captures
MAX_IMAGES = 10             # max images in folder
OUTPUT_DIR = "temp_frames"  # folder to store frames
CAMERA_INDEX = 0            # 0 usually = default laptop camera
# =====================

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for f in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: could not delete {file_path}: {e}")

    # Queue to track saved filenames in order (oldest -> newest)
    saved_files = deque()

    # Open the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # For timing the captures
    last_capture_time = 0.0

    print("Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        # Show the live video
        cv2.imshow("Live Camera", frame)

        # Check if it's time to capture a frame
        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            last_capture_time = current_time

            # Create a filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            millis = int((current_time - int(current_time)) * 1000)
            filename = f"frame_{timestamp}_{millis:03d}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Save the image
            cv2.imwrite(filepath, frame)
            saved_files.append(filepath)
            print(f"Saved: {filepath}")

            # If we exceed MAX_IMAGES, delete the oldest one
            if len(saved_files) > MAX_IMAGES:
                oldest_file = saved_files.popleft()
                if os.path.exists(oldest_file):
                    try:
                        os.remove(oldest_file)
                        print(f"Deleted oldest: {oldest_file}")
                    except Exception as e:
                        print(f"Warning: could not delete {oldest_file}: {e}")

        # Handle quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
