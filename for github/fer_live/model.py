# fer_live/model.py
import numpy as np
import cv2

CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]

class EmotionModel:
    """Thin wrapper around HSEmotion to always run on CPU and return {class: prob}."""
    def __init__(self, model_name="enet_b2_7", device="cpu"):
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.model = HSEmotionRecognizer(model_name=model_name, device=device)

        # try to discover class order exposed by the model
        self.class_order = None
        for attr in ("emotions", "class_names", "classes", "idx_to_class"):
            v = getattr(self.model, attr, None)
            if isinstance(v, (list, tuple)) and len(v) >= 7:
                self.class_order = list(v); break

    def _scores_to_dict(self, raw):
        if isinstance(raw, dict):
            # ensure all expected keys exist
            return {k: float(raw.get(k, 0.0)) for k in CLASSES}
        arr = np.asarray(raw).reshape(-1).astype(float)
        order = self.class_order if (self.class_order and len(self.class_order) == len(arr)) else CLASSES[:len(arr)]
        return {name: float(arr[i]) for i, name in enumerate(order)}

    def predict_bgr(self, face_bgr):
        """Input: aligned 224x224 BGR. Output: dict {class: prob}."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        _top, raw = self.model.predict_emotions(face_rgb, logits=False)
        return self._scores_to_dict(raw)
