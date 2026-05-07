import cv2 as cv
import numpy as np
from PIL import Image

cap = cv.VideoCapture(0)

yellow = [0, 255, 255]

def util(color):
    c = np.uint8([[color]])
    hsv_c = cv.cvtColor(c, cv.COLOR_BGR2HSV)
    hue = hsv_c[0][0][0]
    if hue >= 165:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

while True:
    ret, frame = cap.read()
    hsv_rec = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # frame = cv.flip(frame, 1)

    lowerLimit, upperLimit = util(yellow)
    mask = cv.inRange(hsv_rec, lowerLimit, upperLimit)
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox:
        x1, y1, x2, y2 = bbox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    

    if not ret:
        break 

    cv.imshow("Webcam", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()             
cv.destroyAllWindows()
cv.waitKey(1)