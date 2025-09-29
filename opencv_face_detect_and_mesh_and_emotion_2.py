# fer_live_pretrained.py
# Real-time emotion recognition (CPU-only) with:
# - MediaPipe Face Mesh (iris landmarks) + OpenCV alignment/UI
# - HSEmotion (EfficientNet-B2) pretrained on AffectNet (7 classes)
# Includes:
#   * torch.load CPU patch (handles CUDA-saved checkpoints on CPU)
#   * TIMM EfficientNet allowlist + small compatibility hotfix
#   * Robust infer_scores that accepts dict or numpy array from hsemotion

# -------------------- Torch CPU-only loader & allowlist --------------------
import torch
_orig_load = torch.load
def _cpu_load(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")   # avoid CUDA deserialization on CPU machines
    kwargs.setdefault("weights_only", False)   # trusted checkpoint
    return _orig_load(*args, **kwargs)
torch.load = _cpu_load

# Allowlist TIMM EfficientNet for PyTorch safe unpickler (future-proof for >=2.6)
try:
    from torch.serialization import add_safe_globals
    from timm.models.efficientnet import EfficientNet
    add_safe_globals([EfficientNet])
except Exception:
    pass

# TIMM compatibility hotfix (prevents AttributeError on some builds)
try:
    import timm
    import timm.layers  # ensure submodule exists
    try:
        from timm.models._efficientnet_blocks import DepthwiseSeparableConv
        if not hasattr(DepthwiseSeparableConv, "conv_s2d"):
            DepthwiseSeparableConv.conv_s2d = None
    except Exception:
        pass
except Exception:
    pass

# -------------------- Standard imports --------------------
import cv2, time, numpy as np, mediapipe as mp
from hsemotion.facial_emotions import HSEmotionRecognizer

# -------------------- Config --------------------
CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]
OUT_SIZE = 224                 # aligned face crop size
MIN_FACE_SIDE = 100            # require at least this size to infer
REQ_W, REQ_H = 1280, 720       # request 720p
PANEL_W = 300                  # right-side panel width
FULLSCREEN_AT_START = True

# -------------------- MediaPipe Face Mesh (for alignment) --------------------
mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,           # enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_IRIS, RIGHT_IRIS = 468, 473  # iris centers (with refine_landmarks=True)

def align_eyes(frame_bgr, lm, out=OUT_SIZE):
    """Eye-level alignment via iris centers → out×out BGR crop (or None)."""
    H, W = frame_bgr.shape[:2]
    try:
        lc = np.array([lm.landmark[LEFT_IRIS].x*W,  lm.landmark[LEFT_IRIS].y*H], dtype=np.float32)
        rc = np.array([lm.landmark[RIGHT_IRIS].x*W, lm.landmark[RIGHT_IRIS].y*H], dtype=np.float32)
    except Exception:
        return None
    ex, ey, dist = 0.35, 0.35, 0.30*out
    dstL = np.array([ex*out - dist/2, ey*out], dtype=np.float32)
    dstR = np.array([ex*out + dist/2, ey*out], dtype=np.float32)
    M = cv2.estimateAffinePartial2D(
        np.vstack([lc, rc]).reshape(-1,1,2),
        np.vstack([dstL, dstR]).reshape(-1,1,2),
        method=cv2.LMEDS
    )[0]
    if M is None:
        return None
    return cv2.warpAffine(frame_bgr, M, (out, out), flags=cv2.INTER_LINEAR)

# -------------------- UI helpers --------------------
def draw_side_panel(h, lines, scores, fps, headline):
    panel = np.full((h, PANEL_W, 3), (24,24,24), dtype=np.uint8)
    pad = 14
    y = pad + 4
    cv2.putText(panel, "Emotions", (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2, cv2.LINE_AA)
    y += 28
    if lines:
        for ln in lines[:8]:
            cv2.putText(panel, ln[:36], (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,255,255), 1, cv2.LINE_AA)
            y += 20
        y += 6
    if scores:
        bar_h, gap = 22, 8
        bar_bw = PANEL_W - 2*pad - 60
        for name, prob in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            pct = int(round(prob*100))
            cv2.putText(panel, f"{name:>8}", (pad, y + bar_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
            x0 = pad + 70
            cv2.rectangle(panel, (x0, y), (x0 + bar_bw, y + bar_h), (60,60,60), -1)
            w_fill = int(bar_bw * max(0.0, min(1.0, prob)))
            cv2.rectangle(panel, (x0, y), (x0 + w_fill, y + bar_h), (0,180,255), -1)
            cv2.putText(panel, f"{pct:3d}%", (x0 + bar_bw + 6, y + bar_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
            y += bar_h + gap
    cv2.line(panel, (pad, h - 60), (PANEL_W - pad, h - 60), (80,80,80), 1)
    cv2.putText(panel, headline, (pad, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(panel, f"FPS {fps:.1f}", (pad, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)
    return panel

def concat_with_panel(frame, panel):
    H, W = frame.shape[:2]
    canvas = np.zeros((H, W + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :W] = frame
    canvas[:, W:] = panel
    return canvas

# -------------------- Pretrained FER model (HSEmotion on CPU) --------------------
fer = HSEmotionRecognizer(model_name='enet_b2_7', device='cpu')

def _scores_to_dict(raw, fer_obj):
    """Normalize hsemotion outputs (dict OR numpy array) to {class: prob}."""
    if isinstance(raw, dict):
        return {k: float(raw.get(k, 0.0)) for k in CLASSES}

    arr = np.asarray(raw).reshape(-1).astype(float)
    # Try to read class order from the model object
    order = None
    for attr in ("emotions", "class_names", "classes", "idx_to_class"):
        v = getattr(fer_obj, attr, None)
        if isinstance(v, (list, tuple)) and len(v) == len(arr):
            order = list(v); break
    if order is None or len(order) != len(arr):
        order = CLASSES[:len(arr)]
    return {name: float(arr[i]) for i, name in enumerate(order)}

def infer_scores(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    _top, scores_raw = fer.predict_emotions(face_rgb, logits=False)
    return _scores_to_dict(scores_raw, fer)

# -------------------- Live loop --------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)

    WIN = "Emotion Live (Pretrained, CPU)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if FULLSCREEN_AT_START else cv2.WINDOW_NORMAL)

    # show versions once for sanity
    try:
        import timm as _tm; import torch as _tc
        sticky_lines = [f"torch {_tc.__version__}", f"timm  {_tm.__version__}", "hsemotion ok"]
    except Exception as e:
        sticky_lines = [f"env err: {type(e).__name__}", str(e)[:34]]

    fps_t0, fps_frames, fps = time.time(), 0, 0.0
    headline = "..."

    while True:
        ok, frame = cap.read()
        if not ok: break
        fps_frames += 1
        t = time.time()
        if t - fps_t0 > 0.5:
            fps = fps_frames / (t - fps_t0); fps_frames = 0; fps_t0 = t

        # Landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
        res = mesh.process(rgb); rgb.flags.writeable = True

        lines, scores = sticky_lines, None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            xs = [int(p.x * frame.shape[1]) for p in lm.landmark]
            ys = [int(p.y * frame.shape[0]) for p in lm.landmark]
            x1, x2 = max(0, min(xs)), min(frame.shape[1]-1, max(xs))
            y1, y2 = max(0, min(ys)), min(frame.shape[0]-1, max(ys))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            aligned = align_eyes(frame, lm, OUT_SIZE)
            if aligned is not None and min(aligned.shape[:2]) >= MIN_FACE_SIDE:
                try:
                    scores = infer_scores(aligned)
                    top = max(scores.items(), key=lambda kv: kv[1])
                    headline = f"{top[0].upper()} {top[1]*100:.0f}%"
                    lines = ["pretrained (cpu)"]
                except Exception as e:
                    msg = str(e)
                    lines = [f"infer error: {type(e).__name__}", msg[:36], msg[36:72], msg[72:108]]
            else:
                lines = ["face too small/align fail"]
        else:
            lines = ["no face"] + sticky_lines

        panel = draw_side_panel(frame.shape[0], lines, scores, fps, headline)
        canvas = concat_with_panel(frame, panel)
        cv2.imshow(WIN, canvas)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        if k == ord('f'):
            is_full = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
