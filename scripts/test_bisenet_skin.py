import os, sys
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
print(root)

sys.path.append(str(root))

from src.parsing.bisenet_parser import ZLLBiSeNetParser

CKPT = str(root / "models" / "79999_iter.pth")
IMG  = "D:/filter_asian/outputs/23000/23905.png"  # 샘플 1장
OUT  = str(root / "outputs" / "previews" / "test_skin_overlay.png")

parser = ZLLBiSeNetParser(ckpt_path=CKPT, input_size=512, device="cuda", skin_idx=1, debug_hist=True)

img = cv2.imread(IMG)
pm  = parser.parse(img)
skin = parser.skin_mask(pm)

# 오버레이 저장
overlay = img.copy()
cnts, _ = cv2.findContours((skin>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, cnts, -1, (0,255,0), 1)
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(OUT, overlay)
print("[OK] wrote", OUT)
