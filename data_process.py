from mtcnn import MTCNN
import cv2
import os
from moviepy.editor import *
import pandas as pd
import shutil
import re
import unicodedata

def extract_best_frame(video_path, image_output_path):
    try:
        # Load the video
        video_clip = VideoFileClip(video_path)
        duration = video_clip.duration  # Duration in seconds
        # Initialize MTCNN detector
        detector = MTCNN()
        best_face = None
        best_frame = None
        # Iterate over frames at 0.5 second intervals
        for t in range(0, int(duration * 2), 1):
            frame = video_clip.get_frame(t / 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(frame_rgb)
            if detections:
                largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                if best_face is None or largest_face['box'][2] * largest_face['box'][3] > best_face['box'][2] * best_face['box'][3]:
                    best_face = largest_face
                    best_frame = frame
        if best_face is not None:
            x, y, width, height = best_face['box']
            x, y = max(0, x), max(0, y)
            face = best_frame[y:y + height, x:x + width]
            face_resized = cv2.resize(face, (128, 128))
            cv2.imwrite(image_output_path, face_resized)
            print(f"[INFO] Best frame extracted and saved to {image_output_path}")
            return True
        else:
            print(f"[INFO] No faces detected in {video_path}")
            return False
    except Exception as e:
        print(f"[ERROR] Error extracting best frame from {video_path}: {e}")
        return False

def extract_and_save_faces(image_path, output_path):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Cannot read image: {image_path}")
            return False

        # Convert the image to RGB (MTCNN expects RGB images)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize MTCNN detector
        detector = MTCNN()

        # Detect faces
        detections = detector.detect_faces(image_rgb)
        if not detections:
            print(f"[INFO] No faces detected in {image_path}")
            return False

        # Select the largest face (if multiple are detected)
        largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])

        # Extract the face region
        x, y, width, height = largest_face['box']
        x, y = max(0, x), max(0, y)  # Ensure coordinates are within bounds
        face = image[y:y + height, x:x + width]

        # Resize face to a fixed size (e.g., 128x128)
        face_resized = cv2.resize(face, (128, 128))

        # Save the extracted face
        cv2.imwrite(output_path, face_resized)
        #print(f"[INFO] Extracted face saved to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not process image {image_path}: {e}")
        return False


def process_images_to_faces(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Check if the image already exists
            if(os.path.exists(os.path.join(output_folder, filename))):
                print(f"[INFO] Face already extracted for {filename}. Skipping...")
                continue
            image_path = os.path.join(image_folder, filename)
            output_path = os.path.join(output_folder, filename)
            extract_and_save_faces(image_path, output_path)

def create_copy_of_image(image_path, output_folder, new_image_name):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, new_image_name)
    if not os.path.exists(image_path):
        print(f"[ERROR] Image {image_path} does not exist")
        return None
    ## make the copy of the image
    shutil.copy(image_path, output_path)
    return output_path

def sanitize_filename(filename):
    # Normalize to remove accents and special characters
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename)


def class_images(images_folder, csv_path):
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        for row in df.itertuples():
            image_name = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.jpg"
            image_new_path = create_copy_of_image(os.path.join(images_folder, image_name), os.path.join(images_folder, row.Emotion), sanitize_filename(f"{row.Utterance}.jpg"))
            print(f"[INFO] Image {image_name} moved to {image_new_path}")
    except Exception as e:
        print(f"[ERROR] Error classifying images: {e}")
        
        
def convert_images_to_grayscale(images_folder):
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Cannot read image: {image_path}")
                continue
            # convert to graysscale but keep 3 channels
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.merge([gray_image, gray_image, gray_image])
            cv2.imwrite(image_path, rgb_image)
            print(f"[INFO] Converted image {filename} to grayscale")
    return True
        
if __name__ == "__main__":
    convert_images_to_grayscale("Data/train_val_multimodal")
