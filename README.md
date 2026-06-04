# Computer Vision Projects

A growing library of computer vision experiments I've built to actually learn the thing — not follow a tutorial, but pick a problem I genuinely wanted to solve and figure it out from scratch. Each project lives in its own folder with its own README.

---

## Projects

### [PinchToHeart](./PinchToHeart/)

A real-time gesture recognition app that detects when you pinch your fingers together and automatically fires a ❤️ to someone over iMessage. No button. No click. Just pinch.

It uses MediaPipe's Hand Landmark Task API to track 21 3D landmarks per hand on every webcam frame. The gesture detection is depth-aware — it checks the Z-axis of two specific landmarks to confirm an actual pinch rather than an accidental near-miss in 2D. A 5-second cooldown guard makes sure you don't accidentally spam someone.

macOS only (iMessage is macOS-only). Built with OpenCV and MediaPipe.

---

### [Superhero Detector](./Superhero-detector/)

An image classifier that identifies 10 Marvel and DC heroes — Batman, Spider-Man, Iron Man, Thor, Captain America, Wonder Woman, Hulk, Superman, Flash, and Aquaman — at **94% top-1 accuracy**.

I started with a hand-rolled TinyCNN that topped out at 55%. Fine-tuning a pretrained ResNet50 on a custom ~3,900-image dataset I scraped myself pushed it to 94%. That's a 38 percentage point jump from transfer learning alone, which still kind of amazes me.

Served through a Gradio web interface with automatic MPS/CUDA/CPU routing. One command and it's running in the browser.

---

## Tech across projects

- [OpenCV](https://opencv.org/) — webcam capture and frame processing
- [MediaPipe](https://mediapipe.dev/) — hand landmark detection
- [PyTorch](https://pytorch.org/) + [torchvision](https://pytorch.org/vision/) — model training and inference
- [Gradio](https://gradio.app/) — web demo UI
- [icrawler](https://github.com/hellock/icrawler) — automated image dataset collection
