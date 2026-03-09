import cv2
import dlib
import torch

# Window Name (Must match exactly for the X button to work)
WINDOW_NAME = "FYP Setup Test"

print(f"✅ OpenCV: {cv2.__version__} | Dlib: {dlib.__version__} | PyTorch: {torch.__version__}")

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

print("✅ Camera active. Press 'q' or click 'X' to close.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    # 1. Check if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # 2. Check if the 'X' button on the window was clicked
    # (WND_PROP_VISIBLE checks if the window is still open)
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Program closed successfully.")