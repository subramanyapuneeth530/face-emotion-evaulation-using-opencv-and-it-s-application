# main.py
import argparse
import time
import cv2

from fer_live import (
    apply_torch_cpu_safety,
    EmotionModel,     # HSEmotion wrapper
    FaceAligner,     # MediaPipe + iris alignment
    draw_side_panel, concat_with_panel, draw_bbox
)

# ---------- Config ----------
REQ_W, REQ_H = 1280, 720
OUT_SIZE = 224
MIN_FACE_SIDE = 100
FULLSCREEN_AT_START = True

# Map model name -> constructor returning an object with .predict_bgr(face_bgr)->{class:prob}
MODEL_FACTORIES = {
    # HSEmotion variants (easy add: enet_b0_7, enet_b2_7, enet_b4_8, etc.)
    "enet_b0_7": lambda: EmotionModel("enet_b0_7", device="cpu"),
    "enet_b2_7": lambda: EmotionModel("enet_b2_7", device="cpu"),  # default
    # "enet_b4_8": lambda: EmotionModel("enet_b4_8", device="cpu"),

    # Example for a custom adapter in fer_live/adapters (uncomment when you add it):
    # "ferplus_resnet18": lambda: FERPlusResNet18Adapter(weights="path/to.pt"),
    # "mobilenetv3_fer":  lambda: MobileNetV3FERAdapter(weights="path/to.onnx"),
}

def run(model_name: str = "enet_b2_7"):
    apply_torch_cpu_safety()

    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Available: {', '.join(MODEL_FACTORIES.keys())}")

    model = MODEL_FACTORIES[model_name]()
    aligner = FaceAligner(out_size=OUT_SIZE)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)

    win = f"Emotion Live [{model_name}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.setWindowProperty(
        win, cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if FULLSCREEN_AT_START else cv2.WINDOW_NORMAL
    )

    # one-time env line(s)
    try:
        import timm as _tm; import torch as _tc
        sticky_lines = [f"torch {_tc.__version__}", f"timm  {_tm.__version__}", "hsemotion ok"]
    except Exception as e:
        sticky_lines = [f"env err: {type(e).__name__}", str(e)[:34]]

    fps_t0, fps_frames, fps = time.time(), 0, 0.0
    headline, scores = "...", None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # FPS (ema over 0.5s window)
        fps_frames += 1
        t = time.time()
        if t - fps_t0 > 0.5:
            fps = fps_frames / (t - fps_t0)
            fps_frames, fps_t0 = 0, t

        # Landmarks + alignment
        lm, aligned = aligner.align_eyes(frame)

        lines = sticky_lines
        if lm is not None:
            draw_bbox(frame, aligner.bbox_from_landmarks(frame, lm))

            if aligned is not None and min(aligned.shape[:2]) >= MIN_FACE_SIDE:
                try:
                    scores = model.predict_bgr(aligned)
                    top = max(scores.items(), key=lambda kv: kv[1])
                    headline = f"{top[0].upper()} {top[1]*100:.0f}%"
                    lines = [f"pretrained (cpu) • {model_name}"]
                except Exception as e:
                    msg = str(e)
                    lines = [f"infer error: {type(e).__name__}", msg[:36], msg[36:72], msg[72:108]]
            else:
                lines = ["face too small/align fail"]
        else:
            lines = ["no face"] + sticky_lines

        # UI compose & show
        panel = draw_side_panel(frame.shape[0], lines, scores, fps, headline)
        canvas = concat_with_panel(frame, panel)
        cv2.imshow(win, canvas)

        # Keys: Esc/q to quit, f to toggle fullscreen
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        if k == ord('f'):
            is_full = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="enet_b2_7",
        choices=list(MODEL_FACTORIES.keys()),
        help="Which pretrained model to use"
    )
    args = parser.parse_args()
    run(args.model)
