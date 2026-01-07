import cv2
import dlib
import csv
import os
from scipy.spatial import distance as dist

# --- CONFIGURATION ---
PREDICTOR_PATH = "src/shape_predictor_68_face_landmarks.dat"
# Paths updated with your specific filenames
REAL_VIDEO = "data/original_sequences/youtube/c23/videos/585.mp4" 
FAKE_VIDEO = "data/manipulated_sequences/Deepfakes/c23/videos/585_599.mp4"

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def extract_ear_data(video_path, label):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    cap = cv2.VideoCapture(video_path)
    
    ears = []
    print(f"⌛ Analyzing {label} video: {os.path.basename(video_path)}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        # If no face is found in a frame, we use a placeholder or previous value
        if len(faces) == 0:
            ears.append(ears[-1] if len(ears) > 0 else 0.3)
            continue

        for face in faces:
            shape = predictor(gray, face)
            left = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
            avg_ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
            ears.append(avg_ear)

    cap.release()
    return ears

# Process the pair
real_data = extract_ear_data(REAL_VIDEO, "REAL")
fake_data = extract_ear_data(FAKE_VIDEO, "FAKE")

# Save results for Excel graphing
output_file = "blink_comparison_results.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Real_EAR", "Fake_EAR"])
    # Match the lengths in case one video is slightly longer
    for i in range(min(len(real_data), len(fake_data))):
        writer.writerow([i, real_data[i], fake_data[i]])

print(f"\n SUCCESS: Results saved to '{output_file}'")
print("You can now open this file in Excel to create your report graph.")