"""
버킷별 샘플 모자이크 (시각 QA)

실행 명령어 (--key를 wrinkle, port로 바꿔 반복 생성):
python scripts\mosaic_from_list.py --config config\config.yaml ^
  --pass_csv D:\face_parsing\outputs\qc\pass.csv ^
  --out D:\face_parsing\outputs\reports\mosaic_redness.png --key redness --k 4 --per_bucket 24
"""
import argparse
from pathlib import Path
import random, math
import cv2, numpy as np
import pandas as pd
import yaml

def load_img(p): 
    im = cv2.imread(str(p))
    return im

def grid_mosaic(paths, tile=256, cols=8):
    cols = max(1, int(cols))
    rows = math.ceil(len(paths)/cols)
    H, W = rows*tile, cols*tile
    canvas = np.zeros((H,W,3), np.uint8)
    for i, p in enumerate(paths[:rows*cols]):
        r, c = divmod(i, cols)
        im = load_img(p)
        if im is None: continue
        im = cv2.resize(im, (tile,tile), interpolation=cv2.INTER_AREA)
        canvas[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = im
    return canvas

def main(cfg_path, pass_csv, out_png, key="redness", k=4, per_bucket=24):
    with open(cfg_path,'r',encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    out_root = Path(cfg["paths"]["output_root"])
    df = pd.read_csv(pass_csv)

    # 버킷 경계
    qs = df[key].quantile(np.linspace(0,1,k+1)).values
    tiles = []
    for i in range(k):
        lo, hi = qs[i], qs[i+1]
        sub = df[(df[key]>=lo) & (df[key]<=hi)]
        sample = sub.sample(min(per_bucket, len(sub)), random_state=42) if len(sub)>0 else sub
        # aligned 경로로 바꿔서 모자이크 생성
        imgs = [(out_root/"aligned"/rel) for rel in sample["rel_path"].values]
        tiles += imgs

    mosaic = grid_mosaic(tiles, tile=256, cols=8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), mosaic)
    print(f"[OK] wrote {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pass_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="redness")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--per_bucket", type=int, default=24)
    args = ap.parse_args()
    main(args.config, args.pass_csv, args.out, args.key, args.k, args.per_bucket)
