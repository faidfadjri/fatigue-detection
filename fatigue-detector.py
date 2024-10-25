import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance as dist

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

# Batasan EAR untuk mendeteksi mata tertutup
EAR_THRESHOLD = 0.25
# Batasan MAR untuk mendeteksi mulut terbuka (menguap)
MAR_THRESHOLD = 0.7

# Jumlah frame berturut-turut untuk mendeteksi kantuk
TIRED_FRAMES = 10
CONSEC_FRAMES = 20

# Variabel penghitung
frame_counter = 0
yawn_counter = 0

# Inisialisasi detektor wajah dan prediktor landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Landmark untuk mata dan mulut
(left_eye_start, left_eye_end) = (42, 48)
(right_eye_start, right_eye_end) = (36, 42)
(mouth_start, mouth_end) = (48, 68)

# Akses kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke skala abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah di frame
    faces = detector(gray)

    for face in faces:
        # Prediksi landmark wajah
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Dapatkan koordinat mata kiri dan kanan
        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]

        # Dapatkan koordinat mulut
        mouth = shape[mouth_start:mouth_end]

        # Hitung EAR untuk kedua mata
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Hitung MAR untuk mulut
        mar = mouth_aspect_ratio(mouth)

        # Visualisasi mata dan mulut pada frame
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        mouth_hull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

        # Deteksi mata tertutup (EAR di bawah threshold)
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if TIRED_FRAMES <= frame_counter < CONSEC_FRAMES:
                cv2.putText(frame, "FEELING TIRED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            elif frame_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frame_counter = 0

        # Calculate tiredness level as a percentage
        tiredness_level = min((frame_counter / CONSEC_FRAMES) * 100, 100)

        # Display the tiredness level percentage
        cv2.putText(frame, f"Tiredness Level: {tiredness_level:.0f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Deteksi menguap (MAR di atas threshold)
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            cv2.putText(frame, "YAWNING DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            yawn_counter = 0

        # Tampilkan nilai EAR dan MAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Tampilkan frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
