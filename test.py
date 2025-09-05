import cv2
import numpy as np
import dlib
import os
import time
import locale
import subprocess

# Установка локали (опционально)
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

# Подключаем камеру
cap = cv2.VideoCapture(0)

# Загружаем детектор лиц из библиотеки dlib
detector = dlib.get_frontal_face_detector()

# Загружаем модель распознавания лиц
path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(path, 'trainer', 'trainer.yml'))

# Звуковые файлы
error_sound = "1.wav"
normal_sound = "2.wav"
sound_map = {
    1: "sound1.wav",  # Замените пути на свои
    2: "sound2.wav",
    3: "sound3.wav",
    # ... добавьте свои ID и соответствующие им звуки
}
# Переменная для отслеживания времени последнего воспроизведения звука
last_sound_time = 0
error_sound_time = 0
sound_delay = 2  # Задержка в секундах между воспроизведениями звука

def play_sound(sound_file):
    try:
        subprocess.Popen(['aplay', sound_file], stderr=subprocess.PIPE)
    except FileNotFoundError:
        print(f"Звуковой файл не найден: {sound_file}")
    except Exception as e:
        print(f"Ошибка воспроизведения звука: {e}")

while True:
    # Захват кадра с камеры
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Переводим в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = detector(gray)

    # Время текущего кадра
    current_time = time.time()

    for i, face in enumerate(faces, start=1):
        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # ROI лица
        face_roi = gray[y:y1, x:x1]

        # Получаем высоту и ширину ROI
        h, w = face_roi.shape[:2]

        # Проверка размеров ROI лица
        if h > 0 and w > 0:
            # Распознаём лицо
            label, confidence = recognizer.predict(face_roi)
            label_text = f"Unknown"
            if confidence < 70:
                if label in sound_map:
                    label_text = str(label)
                    if (current_time - last_sound_time) >= sound_delay:
                         play_sound(normal_sound)
                         last_sound_time = current_time
                else:
                     label_text = f"Unknown, id={label}"

            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
           if (current_time - error_sound_time) >= sound_delay:
               print("Ошибка: Обнаружена область лица с нулевым размером.")
               play_sound(error_sound)
               error_sound_time = current_time

    # Показываем кадр
    cv2.imshow("Enemy faces", frame)

    # Нажатие 'q' завершает программу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()