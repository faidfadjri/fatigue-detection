import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance as dist
import tkinter as tk
from threading import Thread

# Fungsi untuk menghitung Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Fungsi untuk menghitung Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])  # Jarak vertikal
    B = dist.euclidean(mouth[14], mouth[18])  # Jarak vertikal
    C = dist.euclidean(mouth[12], mouth[16])  # Jarak horizontal
    mar = (A + B) / (2.0 * C)
    return mar

# Batasan EAR dan MAR
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.7
TIRED_FRAMES = 10
CONSEC_FRAMES = 20

# Variabel kontrol
frame_counter = 0
yawn_counter = 0
detection_enabled = False  # Untuk mengontrol apakah deteksi diaktifkan

# Ground truth data (simulated for evaluation purposes)
ground_truth = {
    "frame_index": [],
    "is_drowsy": [],  # 1 for drowsy, 0 for alert
    "is_yawning": []  # 1 for yawning, 0 for not yawning
}

# Inisialisasi detektor wajah dan prediktor landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(left_eye_start, left_eye_end) = (42, 48)
(right_eye_start, right_eye_end) = (36, 42)
(mouth_start, mouth_end) = (48, 68)

def fatigue_detection():
    global detection_enabled, frame_counter, yawn_counter

    # Akses kamera
    cap = cv2.VideoCapture(0)
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Jika deteksi tidak diaktifkan, tampilkan frame tanpa proses
        if not detection_enabled:
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        faces = detector(gray)

        is_drowsy = 0
        is_yawning = 0

        for face in faces:
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = shape[left_eye_start:left_eye_end]
            right_eye = shape[right_eye_start:right_eye_end]
            mouth = shape[mouth_start:mouth_end]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            mouth_hull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if TIRED_FRAMES <= frame_counter < CONSEC_FRAMES:
                    cv2.putText(frame, "FEELING TIRED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    is_drowsy = 1
                elif frame_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    is_drowsy = 1
            else:
                frame_counter = 0

            tiredness_level = min((frame_counter / CONSEC_FRAMES) * 100, 100)
            cv2.putText(frame, f"Tiredness Level: {tiredness_level:.0f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                cv2.putText(frame, "YAWNING DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                is_yawning = 1
            else:
                yawn_counter = 0

            cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Append to ground truth
        ground_truth["frame_index"].append(frame_index)
        ground_truth["is_drowsy"].append(is_drowsy)
        ground_truth["is_yawning"].append(is_yawning)
        frame_index += 1

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print or save ground truth for later analysis
    print("Ground Truth Data:")
    for i in range(len(ground_truth["frame_index"])):
        print(f"Frame {ground_truth['frame_index'][i]}: Drowsy={ground_truth['is_drowsy'][i]}, Yawning={ground_truth['is_yawning'][i]}")

def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    status = "ON" if detection_enabled else "OFF"
    button.config(text=f"Detection: {status}")

# GUI dengan tkinter
root = tk.Tk()
root.title("Fatigue Detector")
root.geometry("200x100")

button = tk.Button(root, text="Detection: OFF", command=toggle_detection)
button.pack(pady=20)

# Jalankan deteksi dalam thread terpisah
thread = Thread(target=fatigue_detection, daemon=True)
thread.start()

root.mainloop()
