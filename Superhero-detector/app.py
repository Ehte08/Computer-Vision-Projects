import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = [
    "aquaman", "batman", "captain_america", "flash",
    "hulk", "iron_man", "spider_man", "superman", "thor", "wonder_woman"
]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(os.path.join(_DIR, "resnet18_superhero.pth"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def predict(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a superhero image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Superhero Detector",
    description="Upload any image and the model will identify which superhero it is.",
)

if __name__ == "__main__":
    demo.launch()
