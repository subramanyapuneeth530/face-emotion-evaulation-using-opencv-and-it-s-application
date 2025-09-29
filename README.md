# face-emotion-evaulation-using-opencv-and-it-s-application
CPU-only real-time emotion app: webcam frames → MediaPipe Face Mesh (iris 468/473) for eye-based alignment → 224×224 crop → pretrained HSEmotion EfficientNet-B2 → seven-class probabilities. OpenCV draws box + side panel with bars, headline, FPS. Torch patch forces CPU loading and handles TIMM quirks.
