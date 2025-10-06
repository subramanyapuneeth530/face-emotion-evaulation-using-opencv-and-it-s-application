# 🎭 Face Emotion Evaluation using OpenCV and HSEmotion

**CPU-only real-time emotion recognition app**  
➡️ Webcam frames → **MediaPipe Face Mesh** (iris alignment) → **224×224 crop** → one of several **HSEmotion pretrained EfficientNet models** → live emotion probabilities with an OpenCV GUI.

Includes:
- CPU-safe PyTorch patch (no CUDA dependency)
- Modular architecture (`fer_live` library)
- Tkinter GUI launcher to choose between models (EffNet-B0/B2/etc.)
- Real-time aligned face inference with per-emotion bar chart + FPS counter

## 🧠 Introduction

This project is a **real-time facial emotion recognition system** designed to run **entirely on CPU**.

It uses:
- **OpenCV** for video I/O and UI visualization  
- **MediaPipe Face Mesh** for high-precision landmark detection (eyes, iris)  
- **HSEmotion** EfficientNet models trained on **AffectNet** (7 emotion classes)

The system performs **iris-based alignment** to normalize face pose, runs inference on the chosen pretrained model, and displays the predicted emotion probabilities in a live side panel.

## ⚙️ Project Structure

emotion_app/
├─ main.py                 ← core live app (accepts --model flag)
├─ launch.py               ← Tkinter GUI launcher (model selection buttons)
│
├─ fer_live/               ← custom Python package (modular code)
│  ├─ __init__.py
│  ├─ utils.py             ← PyTorch CPU-safe loader + TIMM fixes
│  ├─ model.py             ← EmotionModel wrapper (HSEmotion inference)
│  ├─ pipeline.py          ← Face alignment via MediaPipe iris landmarks
│  ├─ gui.py               ← OpenCV-based UI drawing (panel, bbox)
│  └─ adapters/            ← (optional) custom model adapters (FER+, ONNX, etc.)
│
├─ assets/                 ← (optional) icons, sample images
├─ requirements.txt        ← dependencies list
└─ README.md

## 🚀 How It Works — CPU-only safe loading

`fer_live/utils.py` patches PyTorch and TIMM:
- Forces CPU deserialization (`map_location='cpu'`)
- Adds `EfficientNet` to the PyTorch safe unpickler allowlist
- Fixes TIMM `DepthwiseSeparableConv` missing attributes

✅ No CUDA required, compatible across PyTorch 2.x

## 🚀 How It Works — Real-time pipeline

- Captures webcam frames using OpenCV  
- Uses **MediaPipe Face Mesh** with `refine_landmarks=True` for precise iris points  
  (`LEFT_IRIS=468`, `RIGHT_IRIS=473`)
- Computes a similarity transform between current and canonical eye coordinates  
- Aligns and crops a **224×224** face patch for consistent input

## 🚀 How It Works — Model inference

- Supports **HSEmotion** pretrained models:
  - `enet_b0_7`
  - `enet_b2_7`
  - (more can be added easily)
- Each model outputs seven probabilities:
  angry, disgust, fear, happy, sad, surprise, neutral

You can add adapters for other frameworks (e.g., FER+, ONNX, MobileNet).

## 🚀 How It Works — Modular visualization (OpenCV UI)

- Draws face bounding box + side info panel:
  - Emotion bars (%)
  - FPS
  - Model name
  - Status/debug messages

Keys:
- `f` → toggle fullscreen
- `q` / `Esc` → quit

## 🛠️ Requirements

Install dependencies:
pip install -r requirements.txt

`requirements.txt` example:
opencv-python
mediapipe
torch
timm
hsemotion
numpy
tk

## 📝 Usage — Command line

python main.py --model enet_b0_7
# or
python main.py --model enet_b2_7

## 📝 Usage — GUI launcher

python launch.py

A window appears with buttons:
- “EffNet-B0 (7)”
- “EffNet-B2 (7)”

Click one → it opens the live feed UI using that model.

Add new models by editing `MODELS` in `launch.py` and `MODEL_FACTORIES` in `main.py`.

## 🧩 Adding More Models

1) **HSEmotion variants**  
   Add the model name to:
   - `MODEL_FACTORIES` in `main.py`
   - `MODELS` list in `launch.py`

2) **Custom adapters (FER+/ONNX/MobileNet)**  
   Create a file under `fer_live/adapters/` implementing a class with:
   predict_bgr(face_bgr) -> dict {class: prob}

   Register a constructor in `MODEL_FACTORIES` and add a button in `launch.py`.

Tip: Keep returned keys aligned to:
["angry","disgust","fear","happy","sad","surprise","neutral"]

## ⚠️ Notes on Weights & Downloads

Some HSEmotion model URLs may change. If a model fails to load with an HTTP 404:
- Prefer `enet_b0_7` or `enet_b2_7` (known-good)
- Temporarily hide broken entries in `launch.py` and `main.py`
- Optionally pin `hsemotion` to a version that hosted those weights, or load from a local path if supported

## 🧩 Supported Emotions

| Class    | Description               |
|--------- |---------------------------|
| angry    | Anger or frustration      |
| disgust  | Disapproval/disgust       |
| fear     | Fear, anxiety             |
| happy    | Joy, amusement            |
| sad      | Sadness                   |
| surprise | Shock, surprise           |
| neutral  | Calm or relaxed           |

## 💡 Applications

- UX testing and engagement analysis  
- EdTech or telehealth emotion monitoring (with consent)  
- Art installations and interactive games  
- Call center / coaching feedback visualization  
- HCI and research experiments

## 🔮 Future Enhancements

- ONNX Runtime / TensorRT backend for acceleration  
- Multi-face support with per-track alignment  
- Temporal smoothing (EMA) for stability  
- More pretrained models + adapters  
- Export for embedded deployment (OpenCV DNN)

## 🧾 Summary

The app uses **MediaPipe** (iris landmarks) to align faces, feeds a canonical **224×224** crop into a chosen **HSEmotion EfficientNet** model, and displays real-time probabilities in an **OpenCV UI**.

Everything is organized as a reusable library (`fer_live`), and a **Tkinter launcher** lets you select the model before the live session starts.
