# src/utils/viz.py
import cv2
import numpy as np
from pathlib import Path

def _to_color_heat(img01):
    """[0..1] float32 -> uint8 BGR heatmap (JET)."""
    img01 = np.clip(img01.astype(np.float32), 0.0, 1.0)
    u8 = (img01 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)  # BGR

def _overlay(bgr, heat_bgr, alpha=0.45):
    return cv2.addWeighted(bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)

def make_cond_tile(bgr, red_map, wrk_map, por_map, save_path, alpha=0.45, font_scale=0.6):
    """
    bgr: 원본 정렬 얼굴(BGR, HxWx3)
    *_map: [0..1] float32, HxW
    save_path: 저장 경로(.png)
    """
    H, W = bgr.shape[:2]
    # heatmaps
    red_h = _to_color_heat(red_map)
    wrk_h = _to_color_heat(wrk_map)
    por_h = _to_color_heat(por_map)

    # overlays
    red_ov = _overlay(bgr, red_h, alpha)
    wrk_ov = _overlay(bgr, wrk_h, alpha)
    por_ov = _overlay(bgr, por_h, alpha)

    # RGB composite (R=red, G=wrinkle, B=pore)
    comp = np.stack([
        np.clip(red_map, 0, 1),
        np.clip(wrk_map, 0, 1),
        np.clip(por_map, 0, 1)
    ], axis=2)
    comp = (comp * 255.0).astype(np.uint8)  # RGB
    comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)

    # 타일 (2 x 3): [original, composite, redness overlay] / [wrinkle overlay, pore overlay, redness heat only]
    def put_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, cv2.LINE_AA)
        return out

    t1 = put_label(bgr,     "Original")
    t2 = put_label(comp_bgr,"Composite  (R=Redness, G=Wrinkle, B=Pore)")
    t3 = put_label(red_ov,  "Redness Overlay")
    t4 = put_label(wrk_ov,  "Wrinkle Overlay")
    t5 = put_label(por_ov,  "Pore Overlay")
    t6 = put_label(red_h,   "Redness Heat")

    row1 = np.hstack([t1, t2, t3])
    row2 = np.hstack([t4, t5, t6])
    tile = np.vstack([row1, row2])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), tile)
    return save_path
