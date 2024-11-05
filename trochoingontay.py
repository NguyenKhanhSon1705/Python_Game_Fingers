import cv2
import mediapipe as mp
import random
import math
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

withScreen = 1280
height = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, withScreen)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Nguyễn Khánh Sơn
soluonghinhtron = 10
Arrar_hinhtron = []
for _ in range(soluonghinhtron):
    x = random.randint(250, withScreen-10)
    y = random.randint(5, height)
    tocdo = 5
    bankinh = 30
    Arrar_hinhtron.append([x, y, tocdo, bankinh])

def tinh_khoang_cach(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

diemso = 0
# Nguyễn Khánh Sơn

# Trương Minh Sơn
folder_path = "Fingers"
finger_images = {}
for i in range(5):
    image_path = os.path.join(folder_path, f"{i}.png")
    finger_images[i] = cv2.imread(image_path)
# Trương Minh Sơn

# https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Không thể mở camera")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
        # Phạm Lê Quân
        fingerCount = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label
                handLandmarks = []

                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])

                # Kiểm tra ngón cái
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount += 1
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount += 1

                # Kiểm tra các ngón khác
                if handLandmarks[8][1] < handLandmarks[6][1]:    # Ngón trỏ
                    fingerCount += 1
                if handLandmarks[12][1] < handLandmarks[10][1]:  # Ngón giữa
                    fingerCount += 1
                if handLandmarks[16][1] < handLandmarks[14][1]:  # Ngón áp út
                    fingerCount += 1
                if handLandmarks[20][1] < handLandmarks[18][1]:  # Ngón út
                    fingerCount += 1
        cv2.putText(image, f"Ngon tay: {fingerCount}", (0,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            # Phạm Lê Quân
               
               
                # Vẽ bàn tay
                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())

        # Trương Minh Sơn
        # Hiển thị ảnh tương ứng với số ngón tay phan mson
        if fingerCount in finger_images:
            finger_image = finger_images[fingerCount]
            if finger_image is not None:
                h, w, _ = finger_image.shape
                image[0:h, 0:w] = finger_image
         # Trương Minh Sơn


        # Nguyễn Khánh Sơn
        cv2.line(image, (200,0), (200,height) , (255,255,255),2, cv2.LINE_AA)
        
        for hinhtron in Arrar_hinhtron:
            x, y, tocdo, bankinh = hinhtron
            y += tocdo
            if y - bankinh > height:
                y = random.randint(-20, 0)
                x = random.randint(40, withScreen - 40)

            cv2.circle(image, (x, y), bankinh, (0, 111, 255), -1)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    finger_x = int(hand_landmarks.landmark[8].x * withScreen)
                    finger_y = int(hand_landmarks.landmark[8].y * height)

                    if tinh_khoang_cach(finger_x, finger_y, x, y) <= bankinh:
                        y = random.randint(-100, 0)
                        x = random.randint(0, withScreen + 200)
                        diemso += 1
                        print("Điểm:", diemso)
                        
            hinhtron[1] = y

        cv2.putText(image, f"Diem so: {diemso}", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,111, 0),4)
        # Nguyễn Khánh Sơn
        # cv2.imshow('demngontay', cv2.flip(image,1))
        cv2.imshow('demngontay', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
