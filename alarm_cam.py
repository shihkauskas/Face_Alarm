# pip install opencv-python numpy dlib

import cv2
import numpy as np
import dlib
import os
import time
import locale

# Установка локали
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

# Подключаем камеру (0 - стандартная веб-камера)
cap = cv2.VideoCapture(0)

# Загружаем детектор лиц из библиотеки dlib
detector = dlib.get_frontal_face_detector()

# Файл со звуком
sound_file = "2.wav"  # Замените на путь к вашему файлу

# Переменная для отслеживания времени последнего воспроизведения звука
last_sound_time = 0
sound_delay = 2  # Задержка в секундах между воспроизведениями звука

while True:
    # Захват кадра с камеры
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Отражаем изображение по горизонтали

    # Переводим в оттенки серого (ускоряет обработку)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = detector(gray)

    # Счётчик лиц
    face_count = len(faces)

    # Если есть лица и прошло достаточно времени с последнего воспроизведения звука
    current_time = time.time()
    if faces and (current_time - last_sound_time) >= sound_delay:
        if os.path.exists(sound_file):
            try:
                os.system(f'aplay "{sound_file}" &')
                last_sound_time = current_time  # Обновляем время последнего воспроизведения
            except Exception as e:
                print(f"Ошибка воспроизведения звука: {e}")
        else:
            print("Звуковой файл не найден: " + sound_file)

    for i, face in enumerate(faces, start=1):
        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0),
                      2)  # Рисуем рамку вокруг лица
        text = f"Enemy {i}"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Подписываем лица


    # Показываем кадр с обнаруженными лицами
    cv2.imshow("Enemy faces", frame)

    # Нажатие 'q' завершает программу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()