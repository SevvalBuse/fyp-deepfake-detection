import cv2
import dlib
import numpy as np

# Initialize Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('src/shape_predictor_68_face_landmarks.dat')

def draw_roi(frame, points, color=(0, 255, 0)):
    """Draws a bounding box around specific landmark points."""
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    cv2.rectangle(frame, (min(x_coords), min(y_coords)), 
                  (max(x_coords), max(y_coords)), color, 2)

# TEST ON ONE OF YOUR NEW DARK VIDEOS (e.g., ID 031 or 111)
video_path = "data/original_sequences/youtube/c23/videos/031.mp4" 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        
        # Define ROIs based on landmark points
        forehead_points = [shape.part(i) for i in range(17, 27)]
        left_cheek = [shape.part(i) for i in [1, 2, 3, 31, 41]]
        right_cheek = [shape.part(i) for i in [13, 14, 15, 35, 46]]

        # Draw the boxes
        draw_roi(frame, forehead_points, (255, 0, 0)) # Blue for Forehead
        draw_roi(frame, left_cheek, (0, 255, 0))    # Green for Cheeks
        draw_roi(frame, right_cheek, (0, 255, 0))

    cv2.imshow("Verifying Pulse Extraction Zones", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()