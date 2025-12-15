import cv2 as cv
import mediapipe as mp
import time
import mac_imessage

def send_heart():
    text = '❤️'
    saiya = '+1-646-986-6723'
    mac_imessage.send_imessage(message=text, phone_number=saiya)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(0)

last_log_time = 0
LOG_COOLDOWN = 2  # time before repeating the action, so it's not spammy
DIST_THRESHOLD = 30  # distance threshold, you can make this smaller or bigger

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # get landmarks 3 and 7 (indx finger and thumb tip bases)
        lm3 = hand_landmarks.landmark[3]
        lm7 = hand_landmarks.landmark[7]

        x3, y3 = int(lm3.x * w), int(lm3.y * h)
        x7, y7 = int(lm7.x * w), int(lm7.y * h)

        # highlight them
        cv.circle(frame, (x3, y3), 6, (0, 0, 255), -1)
        cv.circle(frame, (x7, y7), 6, (0, 255, 0), -1)

        # get the distance
        dx = x7 - x3
        dy = y7 - y3
        dist = (dx**2 + dy**2) ** 0.5

        cv.putText(frame, f"d={int(dist)}", (10, 40),
                   cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # send message if they get close enough
        now = time.time()
        if dist < DIST_THRESHOLD and (now - last_log_time) > LOG_COOLDOWN:
            send_heart()
            last_log_time = now

        

    cv.imshow("Hand", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
