# PinchToHeart

A real-time computer vision app that detects a pinch gesture via webcam and automatically fires a heart emoji to someone you love over iMessage — no button press required.

Make a pinching gesture with your thumb and index finger. That's it.

---

## Highlights

- **Real-time hand tracking** — MediaPipe's Hand Landmark Task API runs inference on every webcam frame, detecting 21 3D landmarks per hand
- **Depth-aware gesture logic** — uses the Z-axis of landmarks 3 and 7 to confirm thumb-over-finger orientation, eliminating false positives from open-hand positions that happen to be close in 2D
- **iMessage automation** — sends a native macOS iMessage via `mac-imessage`; no third-party messaging service or API key required
- **Cooldown guard** — 5-second debounce prevents accidental message spam

---

## How It Works

1. OpenCV captures a mirrored webcam feed at native frame rate
2. Each frame is converted to RGB and passed to MediaPipe `HandLandmarker`
3. Landmark 3 (thumb IP joint) and landmark 7 (index finger PIP joint) are extracted
4. **2D proximity check** — Euclidean pixel distance between the two points < 30px
5. **3D depth check** — `z3 > z7` confirms the thumb is physically behind the index finger (a true pinch, not an accidental overlap)
6. If both conditions hold and 5 seconds have passed since the last send, a ❤️ is dispatched via iMessage

---

## Requirements

- macOS (iMessage integration is macOS-only)
- Python 3.9+
- A webcam
- The `hand_landmarker.task` model file (included in this repo)

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Open `facetime_text.py` and set the recipient's phone number:

```python
saiya = '+1-XXX-XXX-XXXX'
```

2. Run the app:

```bash
python facetime_text.py
```

3. Press **Q** to quit.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DIST_THRESHOLD` | `30` | Pixel distance (px) between landmarks 3 and 7 to trigger a pinch |
| `LOG_COOLDOWN` | `5` | Seconds between messages |

Increase `DIST_THRESHOLD` if the gesture isn't triggering; decrease it if it fires too easily.

---

## Project Structure

```
PinchToHeart/
├── facetime_text.py      # Main app — real-time gesture detection + iMessage send
├── textmsg.py            # Standalone test script for sending a message manually
└── hand_landmarker.task  # MediaPipe Hand Landmark model (bundled)
```

---

## Tech Stack

- [OpenCV](https://opencv.org/) — webcam capture, frame flipping, and overlay rendering
- [MediaPipe](https://mediapipe.dev/) — 21-point 3D hand landmark detection (Task API)
- [mac-imessage](https://pypi.org/project/mac-imessage/) — native iMessage automation on macOS
