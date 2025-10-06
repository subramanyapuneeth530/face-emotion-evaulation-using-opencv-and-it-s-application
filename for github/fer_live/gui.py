# fer_live/gui.py
import cv2, numpy as np

PANEL_W = 300

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

def draw_bbox(frame, bbox):
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
