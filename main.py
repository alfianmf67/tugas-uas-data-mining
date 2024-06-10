import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Muat model yang telah dilatih
model = load_model('gender_detection_model.h5')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop wajah dari gambar
        face = gray[y:y+h, x:x+w]

        # Resize wajah ke ukuran 150x150
        face = cv2.resize(face, (150, 150))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

        # Normalisasi wajah
        face = face / 255.0

        # Expand dimensi wajah untuk input model
        face = np.expand_dims(face, axis=0)

        # Prediksi gender menggunakan model
        prediction = model.predict(face)

        # Tampilkan hasil prediksi
        if prediction > 0.5:
            gender = 'Male'
        else:
            gender = 'Female'

        # Tampilkan hasil prediksi pada gambar
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan gambar
    cv2.imshow('Gender Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()