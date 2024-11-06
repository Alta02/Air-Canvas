import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Setup Video Capture
cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = None, None
pen_color = (0, 255, 0)  # Warna default IJO

# Definisi warna dan teks
colors = {
    "red": ((0, 0, 255), "Strawberry"),
    "blue": ((255, 0, 0), "Blueberry"),
    "yellow": ((0, 255, 255), "Banana"),
    "green": ((0, 255, 0), "Burgir"),
    "eraser": ((0, 0, 0), "Hapus")  
}
rem_spacing = 30  

# Fungsi untuk menggambar kotak warna di tengah atas
def draw_color_boxes(frame):
    h, w, _ = frame.shape
    box_size = 70
    start_x = w // 2 - (len(colors) * (box_size + rem_spacing)) // 2  # Mulai dari tengah dan geser ke kiri
    
    for i, (color_name, (color, text)) in enumerate(colors.items()):
        # Koordinat kotak
        x1 = start_x + i * (box_size + rem_spacing)
        y1 = 10
        x2, y2 = x1 + box_size, y1 + box_size
        
        # Gambar kotak warna dengan border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Tambahkan teks di dalam kotak
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (x1 + 5, y1 + box_size // 2 + 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Fungsi untuk mendeteksi apakah hanya jari telunjuk yang tegak lurus atau bersama jari tengah
def is_index_up(hand_landmarks):
    # Ambil posisi jari telunjuk dan jari tengah
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Periksa posisi jari (tegak lurus saat ujung lebih tinggi dari pangkalnya di sumbu y)
    is_index_straight = index_finger_tip.y < index_finger_mcp.y
    is_middle_straight = middle_finger_tip.y < middle_finger_mcp.y

    return is_index_straight, is_middle_straight

drawing = False  # Status untuk mulai atau berhenti menggambar

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirroring untuk tampilan seperti cermin
    h, w, _ = frame.shape

    # Konversi ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Inisialisasi canvas jika belum ada
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Gambar kotak warna di tengah atas
    draw_color_boxes(frame)

    # Jika ada tangan yang terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dapatkan koordinat jari telunjuk (index finger tip)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Periksa apakah hanya jari telunjuk yang tegak lurus
            is_index_straight, is_middle_straight = is_index_up(hand_landmarks)

            # Jika hanya jari telunjuk tegak lurus, aktifkan menggambar
            if is_index_straight and not is_middle_straight:
                drawing = True
            # Jika jari telunjuk dan jari tengah tegak lurus, pause menggambar
            elif is_index_straight and is_middle_straight:
                drawing = False
                prev_x, prev_y = None, None  # Reset titik sebelumnya

            # Cek apakah jari berada di area kotak warna
            box_size = 50
            start_x = w // 2 - (len(colors) * (box_size + rem_spacing)) // 2
            if 10 < y < 10 + box_size:
                for i, color_name in enumerate(colors):
                    if start_x + i * (box_size + rem_spacing) < x < start_x + (i + 1) * box_size + i * rem_spacing:
                        pen_color = colors[color_name][0]

            # Gambar hanya ketika dalam mode "menggambar"
            if drawing:
                # Gambar garis pada canvas
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), pen_color, 5)
                prev_x, prev_y = x, y  # Update posisi sebelumnya
            else:
                prev_x, prev_y = None, None  # Reset jika tidak menggambar

            # Tampilkan titik jari telunjuk pada frame
            cv2.circle(frame, (x, y), 10, pen_color, -1)

    else:
        prev_x, prev_y = None, None  # Reset jika tidak ada tangan

    # Gabungkan frame asli dan canvas
    combined = cv2.add(frame, canvas)

    # Tampilkan hasil
    cv2.imshow("Air Canvas coyy", combined)

    # Tekan 'q' untuk keluar, 'c' untuk clear canvas
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas = np.zeros_like(frame)

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()
