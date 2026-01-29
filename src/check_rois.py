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

# TEST ON ONE OF VIDEOS (e.g., ID 031 or 111)
video_path = "data/original_sequences/youtube/c23/videos/241.mp4" 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        
        cheek_down = 4  # pixels (3–6 is fine)

        # ---- ROI CALCULATIONS ----
        face_height = shape.part(8).y - shape.part(27).y
        offset = int(face_height * 0.1)

        forehead_bottom = min(shape.part(19).y, shape.part(24).y) - offset
        forehead_top = forehead_bottom - offset * 2
        forehead_left = shape.part(18).x
        forehead_right = shape.part(25).x

        l_cheek_x = (shape.part(2).x + shape.part(31).x) // 2
        l_cheek_y = ((shape.part(40).y + shape.part(31).y) // 2) + cheek_down


        r_cheek_x = (shape.part(14).x + shape.part(35).x) // 2
        r_cheek_y = ((shape.part(47).y + shape.part(35).y) // 2) + cheek_down


        # ---- DRAW ROIs (MUST BE HERE) ----
        cv2.rectangle(frame,
                      (forehead_left, forehead_top),
                      (forehead_right, forehead_bottom),
                      (255, 0, 0), 2)

        cv2.rectangle(frame,
                      (l_cheek_x - 10, l_cheek_y - 10),
                      (l_cheek_x + 10, l_cheek_y + 10),
                      (0, 255, 0), 2)

        cv2.rectangle(frame,
                      (r_cheek_x - 10, r_cheek_y - 10),
                      (r_cheek_x + 10, r_cheek_y + 10),
                      (0, 255, 0), 2)

    cv2.imshow("Verifying Pulse Extraction Zones", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()


