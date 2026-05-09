# Superhero Detector

**94% top-1 accuracy** across 10 Marvel/DC heroes — built end-to-end from automated web scraping through a fine-tuned ResNet50, served via a real-time Gradio web interface with Apple Silicon (MPS) acceleration.

> Batman · Spider-Man · Iron Man · Thor · Captain America · Wonder Woman · Hulk · Superman · Flash · Aquaman

---

## Highlights

- **Full ML pipeline** — data collection, deduplication, augmentation, training, evaluation, and deployment in a single reproducible notebook
- **Transfer learning at scale** — ImageNet-pretrained ResNet50 fine-tuned on a custom ~3,900-image dataset; a hand-rolled TinyCNN baseline plateaued at ~55%
- **Hardware-aware inference** — automatically routes to Apple MPS, CUDA, or CPU at runtime
- **Interactive demo** — one-command Gradio app returns top-5 predictions with confidence scores

---

## Model Progression

| Model | Test Accuracy | Notes |
|---|---|---|
| TinyCNN (no augmentation) | 55.8% | 3-layer CNN, 8.5M params, 128×128 input |
| TinyCNN + augmentation | 54.8% | RandomCrop · Flip · Jitter · Rotation |
| ResNet18 fine-tuned | 89.4% | ImageNet pretrained, 11.2M params |
| **ResNet50 fine-tuned** | **94.0%** | ImageNet pretrained, 23.5M params — **deployed** |

Transfer learning delivered a **+38 percentage point** lift over the custom baseline.

---

## Architecture & Training

| Stage | Detail |
|---|---|
| Backbone | ResNet50 (ImageNet pretrained) |
| Head | Linear(2048 → 10) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Epochs | 10 |
| Input size | 224 × 224 |
| Batch size | 16 |
| Augmentation | RandomResizedCrop · RandomHorizontalFlip · RandomRotation(15°) · ColorJitter |
| Normalization | ImageNet mean/std `[0.485, 0.456, 0.406]` |

---

## Data Pipeline

1. **Scraping** — `icrawler` + BingImageCrawler pulls ~500 images per hero across 5 targeted search queries each
2. **Deduplication** — MD5 hash filtering removes 134 byte-identical duplicates, leaving **3,894 clean images**
3. **Split** — stratified 80/20 train/test split (3,111 train · 783 test), seeded for reproducibility
4. **Augmentation** — applied on-the-fly during training only; test set is kept clean

---

## Quickstart

```bash
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860), upload any superhero image, and get top-5 predictions with confidence scores.

---

## Project Structure

```
Superhero-detector/
├── app.py                   # Gradio inference server (MPS / CUDA / CPU auto-detect)
├── superhero_cnn.ipynb      # End-to-end notebook: scraping → training → evaluation
├── resnet50_superhero.pth   # Trained model weights (ResNet50, 94.0% accuracy)
├── resnet18_superhero.pth   # Trained model weights (ResNet18, 89.4% accuracy)
├── my_heroes/               # Raw scraped dataset (~3,894 images after dedup)
├── my_heroes_split/         # Train/test split (80/20)
└── requirements.txt
```

---

## Tech Stack

- [PyTorch](https://pytorch.org/) — model training and inference
- [torchvision](https://pytorch.org/vision/) — ResNet pretrained weights and transforms
- [Gradio](https://gradio.app/) — web demo UI
- [icrawler](https://github.com/hellock/icrawler) — automated image dataset collection
- [scikit-learn](https://scikit-learn.org/) — train/test splitting and evaluation metrics
- [Pillow](https://pillow.readthedocs.io/) — image loading and preprocessing
