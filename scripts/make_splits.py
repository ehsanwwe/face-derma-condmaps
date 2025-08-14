"""
학습용 매니페스트 & 분할(train/val)

실행 명령어:
python scripts\make_splits.py --config config\config.yaml ^
  --pass_csv D:\face_parsing\outputs\qc\pass.csv ^
  --out D:\face_parsing\outputs\splits --val_ratio 0.1 --bins 3
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def bucketize(x, bins):
    # 0..bins-1
    q = np.quantile(x, np.linspace(0,1,bins+1))
    idx = np.digitize(x, q[1:-1], right=True)
    return idx

def main(cfg_path, pass_csv, out_dir, val_ratio=0.1, bins=3):
    with open(cfg_path,'r',encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    out_root = Path(cfg["paths"]["output_root"])

    df = pd.read_csv(pass_csv)
    # 3개 점수 버킷으로 단순 층화
    b_wr = bucketize(df["wrinkle"].values, bins)
    b_pr = bucketize(df["pore"].values,    bins)
    b_rd = bucketize(df["redness"].values, bins)
    code  = (b_wr*100 + b_pr*10 + b_rd).astype(int)
    df["bucket"] = code

    # 버킷별로 val_ratio 만큼 샘플
    val_idx = []
    for b in sorted(df["bucket"].unique()):
        sub = df[df["bucket"]==b]
        nval = max(1, int(round(len(sub)*val_ratio)))
        val_idx += sub.sample(nval, random_state=42).index.tolist()
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # 매니페스트 JSONL (image, cond, scores)
    man_path = out/"manifest.jsonl"
    with open(man_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rel = row["rel_path"]
            rec = {
                "image": str(out_root/"aligned"/rel).replace("\\","/"),
                "cond" : str(out_root/"maps/cond"/Path(rel).with_suffix(".npy")).replace("\\","/"),
                "scores": {
                    "wrinkle": float(row["wrinkle"]),
                    "pore": float(row["pore"]),
                    "redness": float(row["redness"])
                },
                "split": row["split"]
            }
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")

    # 텍스트 리스트
    (out/"train.txt").write_text("\n".join(df[df["split"]=="train"]["rel_path"].tolist()), encoding="utf-8")
    (out/"val.txt").write_text("\n".join(df[df["split"]=="val"]["rel_path"].tolist()), encoding="utf-8")

    print(f"[OK] wrote {man_path}, train/val lists at {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pass_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--bins", type=int, default=3)
    args = ap.parse_args()
    main(args.config, args.pass_csv, args.out, args.val_ratio, args.bins)
