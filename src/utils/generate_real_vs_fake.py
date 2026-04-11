"""
Generates a side-by-side real vs fake video frame comparison with landmark overlays.
Uses the same ROI definitions as physio_extractor.py and generate_landmark_overlay.py.
Output: data/report_visuals/real_vs_fake_comparison.png
"""

import cv2
import dlib
import numpy as np
import os

REAL_VIDEO  = "data/audit_set/original_sequences/youtube/c23/videos/075.mp4"
FAKE_VIDEO  = "data/audit_set/manipulated_sequences/Deepfakes/c23/videos/075_977.mp4"
SHAPE_PREDICTOR = "src/shape_predictor_68_face_landmarks.dat"
OUTPUT_PATH = "data/report_visuals/real_vs_fake_comparison.png"

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


def get_best_frame(video_path, max_frames=120):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return None, None

    best_frame = None
    best_shape = None

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:
            face  = max(faces, key=lambda r: (r.right()-r.left())*(r.bottom()-r.top()))
            shape = predictor(gray, face)
            fh    = shape.part(8).y - shape.part(27).y
            if best_frame is None or fh > (best_shape.part(8).y - best_shape.part(27).y):
                best_frame = frame.copy()
                best_shape = shape

    cap.release()
    return best_frame, best_shape


def draw_overlay(frame, shape):
    pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    face_height = shape.part(8).y - shape.part(27).y

    overlay = frame.copy()

    # Forehead ROI (green)
    fh       = max(4, int(face_height * 0.10))
    f_bottom = min(shape.part(19).y, shape.part(24).y) - int(face_height * 0.02)
    f_top    = max(0, f_bottom - fh)
    f_left   = shape.part(18).x
    f_right  = shape.part(25).x
    cv2.rectangle(overlay, (f_left, f_top), (f_right, f_bottom), (0, 220, 0), -1)

    # Cheek ROIs (orange)
    side      = max(4, int(face_height * 0.08))
    cheek_down = int(face_height * 0.02)

    lx = (shape.part(2).x + shape.part(31).x) // 2
    ly = (shape.part(40).y + shape.part(31).y) // 2 + cheek_down
    cv2.rectangle(overlay, (lx - side, ly - side), (lx + side, ly + side), (255, 140, 0), -1)

    rx = (shape.part(14).x + shape.part(35).x) // 2
    ry = (shape.part(47).y + shape.part(35).y) // 2 + cheek_down
    cv2.rectangle(overlay, (rx - side, ry - side), (rx + side, ry + side), (255, 140, 0), -1)

    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # All 68 landmarks (white)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

    # Eye landmarks 36-47 (cyan)
    for i in range(36, 48):
        cv2.circle(frame, pts[i], 4, (0, 255, 255), -1)

    # ROI borders
    cv2.rectangle(frame, (f_left, f_top), (f_right, f_bottom), (0, 220, 0), 2)
    cv2.rectangle(frame, (lx - side, ly - side), (lx + side, ly + side), (255, 140, 0), 2)
    cv2.rectangle(frame, (rx - side, ry - side), (rx + side, ry + side), (255, 140, 0), 2)

    return frame


def add_label_bar(frame, label, color):
    """Add a coloured label bar at the top of the frame."""
    bar_h = 50
    bar = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
    bar[:] = color
    cv2.putText(bar, label,
                (frame.shape[1] // 2 - 60, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, frame])


def run():
    print("Processing real video...")
    real_frame, real_shape = get_best_frame(REAL_VIDEO)
    if real_frame is None:
        print("No face found in real video.")
        return

    print("Processing fake video...")
    fake_frame, fake_shape = get_best_frame(FAKE_VIDEO)
    if fake_frame is None:
        print("No face found in fake video.")
        return

    # Draw overlays
    real_out = draw_overlay(real_frame, real_shape)
    fake_out = draw_overlay(fake_frame, fake_shape)

    # Resize to same height
    target_h = 480
    def resize_to_height(img, h):
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    real_out = resize_to_height(real_out, target_h)
    fake_out = resize_to_height(fake_out, target_h)

    # Add label bars
    real_out = add_label_bar(real_out, "REAL", (34, 139, 34))    # dark green
    fake_out = add_label_bar(fake_out, "FAKE", (34, 34, 180))    # dark red (BGR)

    # Resize to same width
    target_w = min(real_out.shape[1], fake_out.shape[1])
    def resize_to_width(img, w):
        ratio = w / img.shape[1]
        return cv2.resize(img, (w, int(img.shape[0] * ratio)))

    real_out = resize_to_width(real_out, target_w)
    fake_out = resize_to_width(fake_out, target_w)

    # Add divider
    divider = np.zeros((6, target_w, 3), dtype=np.uint8)
    divider[:] = (200, 200, 200)

    combined = np.vstack([real_out, divider, fake_out])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, combined)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
