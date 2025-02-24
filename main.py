import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import logging
import pandas as pd

# Налаштовуємо логування: усі повідомлення записуються у файл "gesture.log"
logging.basicConfig(filename="gesture.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Початкове значення порогу відкритості руки
openness_threshold = 0.6


# Функція для оновлення порогу відкритості через trackbar
def update_threshold(x):
    global openness_threshold
    # Значення trackbar має діапазон від 0 до 1000, тому нормалізуємо його до [0, 1]
    openness_threshold = x / 1000.0


# Функція для обчислення коефіцієнта відкритості руки
def compute_openness(landmarks):
    # Використовуємо координати зап’ястя (landmark 0)
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    # Індекси кінчиків пальців за схемою MediaPipe
    fingertips_indices = [4, 8, 12, 16, 20]
    distances = []  # Список для зберігання відстаней від зап’ястя до кінчиків пальців
    xs = []  # Список x-координат усіх лендмарків
    ys = []  # Список y-координат усіх лендмарків

    # Проходимо по всіх лендмарках та записуємо їх координати
    for lm in landmarks:
        xs.append(lm.x)
        ys.append(lm.y)

    # Обчислюємо відстані від зап’ястя до кожного кінчика пальця
    for idx in fingertips_indices:
        tip = np.array([landmarks[idx].x, landmarks[idx].y])
        distances.append(np.linalg.norm(tip - wrist))

    # Середнє значення відстаней
    avg_distance = np.mean(distances)

    # Визначаємо розмір руки, як максимальну ширину або висоту bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    hand_size = max(max_x - min_x, max_y - min_y)

    # Захищаємося від ділення на нуль
    if hand_size == 0:
        return 0
    # Повертаємо відношення середньої відстані до розміру руки
    return avg_distance / hand_size


# Функція для класифікації жесту за коефіцієнтом відкритості
def classify_gesture_by_openness(landmarks, threshold):
    # Обчислюємо коефіцієнт відкритості
    ratio = compute_openness(landmarks)
    # Обчислюємо "схожість" як відсоток: 100% – коли ratio точно дорівнює threshold,
    # а відхилення зменшують значення
    similarity = max(0, min(100, (1 - abs(ratio - threshold) / threshold) * 100))
    # Якщо коефіцієнт відкритості менший за поріг, вважаємо, що визначено літеру "X"
    if ratio < threshold:
        return "X", True, ratio, similarity
    else:
        return "Not X", False, ratio, similarity


# Ініціалізуємо MediaPipe Hands для розпізнавання рук
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,  # Відстежуємо лише одну руку
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
# Інструменти для малювання лендмарків на зображенні
mp_draw = mp.solutions.drawing_utils

# Відкриваємо вебкамеру
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera open error!")
    exit()

# Отримуємо розміри кадру з камери
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Створюємо вікно для відображення відео
cv2.namedWindow("Video")
# Створюємо trackbar для зміни порогу відкритості
cv2.createTrackbar("Openness Thresh", "Video", int(openness_threshold * 1000), 1000, update_threshold)

# Налаштування для запису відео (за потреби)
record_video = False
video_writer = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек для запису відео
output_video_file = "gesture_output.avi"

# Списки для логування даних
data_log = []  # Дані для загального звіту (час, FPS, жест, коефіцієнт відкритості, схожість)
landmarks_log = []  # Дані для кожного лендмарку (координати)

# Початковий час запуску програми
start_time = time.time()
# Зберігаємо час попереднього кадру для обчислення FPS
prev_frame_time = time.time()

print("Press 'q' to quit, 'r' to start/stop recording, 's' to save data to CSV.")

# Основний цикл обробки кадрів з камери
while True:
    ret, frame = cap.read()  # Зчитуємо кадр з камери
    if not ret:
        continue  # Якщо кадр не зчитано, переходимо до наступного
    frame = cv2.flip(frame, 1)  # Перевертаємо кадр по горизонталі для зручності
    current_time = time.time()
    # Обчислюємо FPS як обернене значення часу між кадрами
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time

    # Конвертуємо зображення з BGR (формат OpenCV) у RGB (формат MediaPipe)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)  # Отримуємо результати розпізнавання руки

    # Ініціалізуємо змінні для зберігання результату
    gesture_label = "No Hand"
    recognized = False
    ratio = 0
    similarity = 0

    # Якщо рука виявлена, обробляємо перший знайдений кадр
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Малюємо лендмарки руки на кадрі
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Записуємо координати кожного лендмарку для подальшого аналізу
            lm_data = {"timestamp": current_time - start_time}
            for idx, lm in enumerate(hand_landmarks.landmark):
                lm_data[f"lm_{idx}_x"] = lm.x
                lm_data[f"lm_{idx}_y"] = lm.y
                lm_data[f"lm_{idx}_z"] = lm.z
            landmarks_log.append(lm_data)

            # Класифікуємо жест, використовуючи коефіцієнт відкритості
            gesture_label, recognized, ratio, similarity = classify_gesture_by_openness(
                hand_landmarks.landmark, openness_threshold
            )
            # Обробляємо лише першу руку
            break

    # Відображаємо різну інформацію на екрані за допомогою cv2.putText:
    # 1. FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    # 2. Коефіцієнт відкритості (ratio)
    cv2.putText(frame, f"Ratio: {ratio:.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    # 3. Відсоток схожості
    cv2.putText(frame, f"Similarity: {similarity:.2f}%", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # 4. Інформація про визначення літери X
    # Якщо схожість більше 70%, вважаємо, що літера "X" визначена
    if similarity > 70:
        cv2.putText(frame, "Detected letter X", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Not detected letter X", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Запис кадру у відеофайл, якщо функція запису увімкнена
    if record_video:
        if video_writer is None:
            video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
        video_writer.write(frame)

    # Відображаємо кадр у вікні "Video"
    cv2.imshow("Video", frame)

    # Обробка натискань клавіш:
    # 'q' для виходу з програми
    # 'r' для старту/зупинки запису відео
    # 's' для збереження даних у CSV файли
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        record_video = not record_video
        if not record_video and video_writer is not None:
            video_writer.release()
            video_writer = None
        print("Recording:", "On" if record_video else "Off")
    elif key & 0xFF == ord('s'):
        # Запис даних про жести у CSV файл
        with open('gesture_data_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'fps', 'gesture_label', 'ratio', 'similarity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data_log:
                writer.writerow(row)
        # Запис координат лендмарків у окремий CSV файл
        with open('landmarks_data_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            if landmarks_log:
                fieldnames = list(landmarks_log[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in landmarks_log:
                    writer.writerow(row)
        print("Data saved to gesture_data_log.csv and landmarks_data_log.csv")
        logging.info("Data saved to gesture_data_log.csv and landmarks_data_log.csv")

    # Формуємо повідомлення для консолі та логів із поточними даними
    console_msg = (f"[LOG] Gesture: {gesture_label} | FPS: {fps:.2f} | "
                   f"Ratio: {ratio:.2f} | Similarity: {similarity:.2f}%")
    print(console_msg)
    logging.info(console_msg)

    # Зберігаємо дані для підсумкового звіту
    data_log.append({
        'timestamp': current_time - start_time,
        'fps': fps,
        'gesture_label': gesture_label,
        'ratio': ratio,
        'similarity': similarity
    })

# Звільняємо ресурси камери та відеозаписувача
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

print("Program finished.")
logging.info("Program finished.")

# Створюємо звіт за допомогою pandas та записуємо його у текстовий файл
df = pd.DataFrame(data_log)
if not df.empty:
    avg_fps = df['fps'].mean()
    avg_ratio = df['ratio'].mean()
    avg_similarity = df['similarity'].mean()
    gesture_counts = df['gesture_label'].value_counts().to_dict()
    total_frames = len(df)

    report_text = (
        f"Gesture recognition report (letter 'X')\n"
        f"========================================\n"
        f"Total frames: {total_frames}\n"
        f"Average FPS: {avg_fps:.2f}\n"
        f"Average ratio: {avg_ratio:.2f}\n"
        f"Average similarity: {avg_similarity:.2f}%\n"
        f"Gesture counts: {gesture_counts}\n"
    )

    with open('gesture_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    logging.info("Report saved to gesture_report.txt")
else:
    print("No data for reporting.")
