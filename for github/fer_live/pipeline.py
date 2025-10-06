# fer_live/pipeline.py
import cv2, numpy as np, mediapipe as mp

LEFT_IRIS, RIGHT_IRIS = 468, 473

class FaceAligner:
    def __init__(self, out_size=224):
        self.out = out_size
        mp_face_mesh = mp.solutions.face_mesh
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
        res = self.mesh.process(rgb); rgb.flags.writeable = True
        return res.multi_face_landmarks[0] if res.multi_face_landmarks else None

    def bbox_from_landmarks(self, frame_bgr, lm):
        xs = [int(p.x * frame_bgr.shape[1]) for p in lm.landmark]
        ys = [int(p.y * frame_bgr.shape[0]) for p in lm.landmark]
        x1, x2 = max(0, min(xs)), min(frame_bgr.shape[1]-1, max(xs))
        y1, y2 = max(0, min(ys)), min(frame_bgr.shape[0]-1, max(ys))
        return (x1, y1, x2, y2)

    def align_eyes(self, frame_bgr):
        """Return (landmarks, aligned_224x224_bgr or None)."""
        lm = self.detect(frame_bgr)
        if lm is None: return None, None

        H, W = frame_bgr.shape[:2]
        try:
            lc = np.array([lm.landmark[LEFT_IRIS].x*W,  lm.landmark[LEFT_IRIS].y*H], dtype=np.float32)
            rc = np.array([lm.landmark[RIGHT_IRIS].x*W, lm.landmark[RIGHT_IRIS].y*H], dtype=np.float32)
        except Exception:
            return lm, None

        ex, ey, dist = 0.35, 0.35, 0.30*self.out
        dstL = np.array([ex*self.out - dist/2, ey*self.out], dtype=np.float32)
        dstR = np.array([ex*self.out + dist/2, ey*self.out], dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(
            np.vstack([lc, rc]).reshape(-1,1,2),
            np.vstack([dstL, dstR]).reshape(-1,1,2),
            method=cv2.LMEDS
        )
        if M is None:
            return lm, None
        aligned = cv2.warpAffine(frame_bgr, M, (self.out, self.out), flags=cv2.INTER_LINEAR)
        return lm, aligned
