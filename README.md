# face-emotion-evaulation-using-opencv-and-it-s-application
CPU-only real-time emotion app: webcam frames → MediaPipe Face Mesh (iris 468/473) for eye-based alignment → 224×224 crop → pretrained HSEmotion EfficientNet-B2 → seven-class probabilities. OpenCV draws box + side panel with bars, headline, FPS. Torch patch forces CPU loading and handles TIMM quirks.

# Introduction
This project is a real-time, CPU-only emotion recognition app built with OpenCV (I/O + UI), MediaPipe Face Mesh (precise facial landmarks), and a pretrained HSEmotion EfficientNet-B2 model (AffectNet, 7 emotions). It aligns the face via iris landmarks, runs the classifier, and renders a side panel with per-emotion probabilities, top label, and FPS.

# How the code works

## 1) Safe CPU-only model loading (PyTorch patch)
- We intercept `torch.load` with a wrapper that **forces `map_location='cpu'`** so checkpoints saved on CUDA deserialize on CPU-only machines.
- We set `weights_only=False` (trusted source) to bypass PyTorch ≥2.6’s strict safe-unpickler default.
- We **allow-list** TIMM’s `EfficientNet` class via `add_safe_globals` so unpickling its modules won’t fail.
- A tiny **TIMM compatibility hotfix** ensures `DepthwiseSeparableConv` has the attribute `conv_s2d` in case your local TIMM build lacks it.

**Why:** Removes CUDA dependency and common version/ABI pitfalls around model loading.

---

## 2) Libraries and configuration
- **OpenCV**: camera capture, drawing, and windowing.
- **MediaPipe Face Mesh**: fast, robust 3D facial landmark regression on CPU.
- **HSEmotion (EfficientNet-B2)**: pretrained 7-class FER head (angry, disgust, fear, happy, sad, surprise, neutral).
- Tunables:
  - `OUT_SIZE=224` (aligned crop size),
  - `MIN_FACE_SIDE=100` (skip too-small faces),
  - `REQ_W×REQ_H=1280×720` (requested capture size),
  - `PANEL_W` (UI panel width),
  - `FULLSCREEN_AT_START`.

---

## 3) Face landmarks (MediaPipe Face Mesh)
- We instantiate `FaceMesh(..., refine_landmarks=True, max_num_faces=1)`.
- Each frame is converted to RGB and passed to `mesh.process()`.
- With `refine_landmarks=True`, two **iris center** landmarks are available:
  - `LEFT_IRIS = 468`, `RIGHT_IRIS = 473`.

**Why:** Iris points are highly stable → precise alignment.

---

## 4) Geometric alignment (eye-based similarity transform)
- We read pixel coordinates of the two iris centers `(l_x,l_y)` and `(r_x,r_y)`.
- We define **canonical eye locations** in the output crop (e.g., x≈35%/y≈35%, with fixed inter-eye distance).
- We estimate a **similarity transform** `M` with `cv2.estimateAffinePartial2D(src_eyes, dst_eyes)`.
  - This solves for rotation + uniform scale + translation (no shear).
- We warp the original frame with `cv2.warpAffine(frame, M, (OUT_SIZE, OUT_SIZE))` to obtain a **224×224 aligned face**.

**Why:** Canonicalized pose (eyes level, consistent scale) greatly improves FER robustness to head tilt/zoom.

---

## 5) Pretrained FER model (HSEmotion EfficientNet-B2)
- We create `HSEmotionRecognizer(model_name='enet_b2_7', device='cpu')`.
- Per aligned face:
  1. Convert BGR → RGB.
  2. Call `predict_emotions(img, logits=False)`.
  3. Receive either:
     - a **dict** `{emotion: prob}`, or
     - a **NumPy array** of class probabilities.

**Why:** EfficientNet-B2 has strong features; the head is trained on AffectNet 7-class setup.

---

## 6) Output normalization (robust across versions)
- The helper `_scores_to_dict(raw, fer)` standardizes outputs:
  - If `raw` is a dict → fill all seven class keys; missing ones become 0.
  - If `raw` is an array → we try to read the model’s class order (e.g., `fer.emotions`, `fer.class_names`).
  - Fallback to our fixed `CLASSES` ordering if the model doesn’t expose it.
- We then compute the **top emotion** and keep the full probability vector for the UI.

---

## 7) Rendering and user interface (OpenCV)
- We draw a quick **bounding box** around the detected landmarks (min/max over all points).
- A right-side **panel** shows:
  - **Status lines** (versions, “no face”, alignment errors, or inference errors with messages).
  - **Bar chart** of all seven emotions (percentages).
  - **Headline** with the top emotion (e.g., `HAPPY 92%`).
  - **FPS** computed over a sliding window (updates every ~0.5s).
- Window starts **fullscreen**; press `f` to toggle, `q`/`Esc` to quit.

---

## 8) Control flow per frame
1. Capture frame from webcam.
2. Run Face Mesh → landmarks (if none, show “no face”).
3. If landmarks exist → compute similarity transform → warp to 224×224.
4. If crop is large enough → run FER model → normalize scores → update UI.
5. Compose `frame + panel`, `imshow`, handle key events.

---

## 9) Performance & accuracy tips
- Keep face reasonably large (≥120 px short side) and evenly lit.
- If FPS dips, lower camera resolution or infer every 2–3 frames.
- Alignment already stabilizes predictions; optional temporal smoothing (EMA) can be added later.

---

## 10) Failure handling
- If model/dep mismatch occurs, the panel shows a readable **exception type and message**.
- TIMM/PyTorch quirks are mitigated by the **CPU patch** and **hotfix**; version info is printed once on startup for quick diagnostics.


# Applications
- UX testing and affective computing dashboards.  
- EdTech/HealthTech engagement monitoring (with consent).  
- Interactive installations, art projects, and games reacting to user emotion.  
- Call-center/coaching tools for post-hoc session review.  
- HCI research prototypes and classroom demos.

# Future endeavours
- Swap backbone to ONNX Runtime (EmotiEff) for zero-PyTorch deps or to leverage GPU (onnxruntime-gpu).  
- Add temporal smoothing (EMA or sliding window) and test-time augmentation for stability.  
- Multi-face support with per-track alignment and asynchronous inference.  
- Personal calibration: tiny logistic head trained on a small user dataset for domain adaptation.  
- Export to ONNX + OpenCV DNN / TensorRT for embedded deployment.  
- Add logging, metrics (macro-F1), and dataset tooling.

# Summary
The app aligns faces with MediaPipe iris landmarks, feeds a canonical crop to a pretrained EfficientNet-B2 FER model, and shows real-time emotion probabilities in an OpenCV UI. A small torch/TIMM patch ensures CPU-safe loading. It’s accurate, fast enough on CPU, and ready to extend (multi-face, smoothing, ONNX, calibration) for production-grade use.
