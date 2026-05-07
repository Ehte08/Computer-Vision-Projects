# Superhero Detector

A computer vision project that identifies superheroes from images or a live webcam feed. Classifies across 10 heroes using a fine-tuned ResNet18 model, achieving **89.4% test accuracy**.

## Heroes
Aquaman · Batman · Captain America · Flash · Hulk · Iron Man · Spider-Man · Superman · Thor · Wonder Woman

## How It Works

1. **Data collection** — ~500 images per hero were scraped from Bing using `icrawler`, deduplicated, and split 80/20 into train/test sets (~3,800 images total).
2. **Training** — A pretrained ResNet18 (ImageNet weights) was fine-tuned with data augmentation (random crop, flip, rotation, color jitter) for 10 epochs using AdamW.
3. **Inference** — At runtime, a pretrained Faster R-CNN detects individual people in the frame, each crop is passed through ResNet18, and results are displayed with bounding boxes.

## Setup

```bash
pip install -r requirements.txt
```

## Running

### Gradio Web App (upload an image)
```bash
cd Superhero-detector
python app.py
```
Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

### Live Webcam Detector (multiple heroes)
Run from **Terminal.app** (not VS Code terminal — OpenCV windows require a native display):
```bash
cd "Superhero-detector"
python webcam_detect.py
```
Press **Q** to quit.

## Project Structure

```
Superhero-detector/
├── app.py                  # Gradio web interface
├── webcam_detect.py        # Live webcam detection with multi-hero support
├── superhero_cnn.ipynb     # Full training notebook (scraping → training → evaluation)
├── resnet18_superhero.pth  # Trained model weights
├── my_heroes/              # Raw scraped dataset
├── my_heroes_split/        # Train/test split dataset
└── requirements.txt
```

## Model Performance

| Model | Test Accuracy |
|---|---|
| TinyCNN (no augmentation) | 55.6% |
| TinyCNN (with augmentation) | 52.6% |
| ResNet18 fine-tuned | **89.4%** |
