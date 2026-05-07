import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np
import os

_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = [
    "aquaman", "batman", "captain_america", "flash",
    "hulk", "iron_man", "spider_man", "superman", "thor", "wonder_woman"
]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

detect_transform = transforms.ToTensor()

def load_classifier():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(os.path.join(_DIR, "resnet18_superhero.pth"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_detector():
    detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    detector.to(DEVICE)
    detector.eval()
    return detector

def detect_people(detector, frame, threshold=0.7):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = detect_transform(rgb).to(DEVICE)
    with torch.no_grad():
        outputs = detector([tensor])[0]

    boxes = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if label == 1 and score >= threshold:  # COCO label 1 = person
            boxes.append(box.cpu().numpy().astype(int))
    return boxes

def classify_crop(classifier, frame, box):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = classify_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(classifier(tensor), dim=1)[0]

    top_idx = probs.argmax().item()
    return CLASS_NAMES[top_idx], float(probs[top_idx])

def draw_results(frame, boxes, results):
    for box, (label, conf) in zip(boxes, results):
        if label is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label.replace('_', ' ').title()}: {conf:.1%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return frame

def main():
    print("Loading classifier...")
    classifier = load_classifier()
    print("Loading person detector...")
    detector = load_detector()
    print(f"Both models loaded on {DEVICE}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    print("Webcam open. Press Q to quit.")
    frame_count = 0
    boxes, results = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if frame_count % 10 == 0:
            boxes = detect_people(detector, frame)
            results = [classify_crop(classifier, frame, box) for box in boxes]

        frame = draw_results(frame, boxes, results)
        cv2.imshow("Superhero Detector", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
