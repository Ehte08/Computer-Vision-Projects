import cv2 as cv
import mediapipe as mp

img = cv.imread("man.jpg")
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections is not None:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            H, W, _ = img.shape
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            img = cv.rectangle(img, (x1, y1), (x1 + w  , y1 + h), (0, 255, 0), 5)

cv.imshow("Face Detection", img)
cv.waitKey(0)
cv.destroyAllWindows()