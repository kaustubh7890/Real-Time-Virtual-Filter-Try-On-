import cv2
import numpy as np
import mediapipe as mp
import os
import math

# Load Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Load hairstyles
def load_hairstyles(folder="/Users/kaustubhbhoir/Documents/my_hairstyles/hairstyle"):
    print("📂 Trying to load from:", folder)
    styles = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".png"):
            path = os.path.join(folder, file)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                styles.append(img)
    return styles

hairstyles = load_hairstyles("/Users/kaustubhbhoir/Documents/my_hairstyles/hairstyle")
current_style_idx = 0

# Overlay with rotation
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return rotated

def overlay_transparent(background, overlay, x, y):
    bg = background.copy()
    h, w = overlay.shape[:2]

    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    return bg

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            left_ear_x = int(landmarks.landmark[234].x * w)
            right_ear_x = int(landmarks.landmark[454].x * w)

            # Center and forehead
            forehead_x = int(landmarks.landmark[10].x * w)
            forehead_y = int(landmarks.landmark[10].y * h)

            # Compute angle from eyes
            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            hair_width = int(abs(right_ear_x - left_ear_x) * 1.5)
            hair_height = int(hair_width * 1.4)

            # Get hairstyle and rotate
            hairstyle = cv2.resize(hairstyles[current_style_idx], (hair_width, hair_height))
            rotated_hairstyle = rotate_image(hairstyle, -angle)

            # Adjust position
            x = left_ear_x - int(hair_width * 0.18)
            y = forehead_y - int(hair_height * 0.4)

            frame = overlay_transparent(frame, rotated_hairstyle, x, y)

    cv2.putText(frame, f"Press 1/2/3 to switch hairstyles | Current: {current_style_idx + 1}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Hairstyle Try-On", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key in [ord('1'), ord('2'), ord('3')]:
        current_style_idx = key - ord('1')
        current_style_idx %= len(hairstyles)

cap.release()
cv2.destroyAllWindows()
