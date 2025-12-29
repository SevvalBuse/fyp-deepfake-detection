import cv2
import dlib
from scipy.spatial import distance as dist

# --- CONFIGURATION ---
PREDICTOR_PATH = "src/shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.25
WINDOW_NAME = "FYP Blink Detector"  # defining the name here to avoid typos

# --- HELPER FUNCTION ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- MAIN SETUP ---
print("Loading Face Landmark Predictor...")
detector = dlib.get_frontal_face_detector()

try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError:
    print(f"ERROR: Could not find '{PREDICTOR_PATH}'")
    exit()

print("Predictor loaded! Opening Webcam...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = []
        for i in range(0, 68):
            landmarks.append((shape.part(i).x, shape.part(i).y))

        leftEye = landmarks[36:42]
        rightEye = landmarks[42:48]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        avgEAR = (leftEAR + rightEAR) / 2.0

        color = (0, 255, 0)
        if avgEAR < EYE_AR_THRESH:
            color = (0, 0, 255)
            cv2.putText(frame, "BLINK DETECTED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, color, -1)
            
        cv2.putText(frame, f"EAR: {avgEAR:.2f}", (300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    # --- CONTROLS ---
    # 1. Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # 2. Click 'X' to quit (Checks if window is still open)
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Blink Detector closed.")