"""
QC 기준 적용해서 "통과/실패" 나누기

실행 명령어 (config.yaml의 qc: 블록에 skin_cov_min: 0.30 확인 및 조정):
python scripts\qc_filter.py --config config\config.yaml ^
  --labels D:\face_parsing\outputs\labels.csv --out D:\face_parsing\outputs\qc --copy
"""

import argparse, shutil
from pathlib import Path
import pandas as pd
import yaml

def main(cfg_path, labels_csv, out_dir, copy=False):
    with open(cfg_path,'r',encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    paths   = cfg["paths"]
    runtime = cfg["runtime"]
    image_root  = Path(paths["image_root"])
    output_root = Path(paths["output_root"])

    df = pd.read_csv(labels_csv)
    # 임계치는 config.yaml의 qc 섹션 사용 + 없으면 suggested_qc.yaml 권장값으로
    qc = cfg.get("qc", {})
    blur_min = float(qc.get("blur_varlap_thresh", 40.0))
    exp_lo_min = float(qc.get("exposure_low_pct", 1.0))
    exp_hi_max = float(qc.get("exposure_high_pct", 250.0))
    skin_cov_min = float(qc.get("skin_cov_min", 0.30))  # ← 새 항목 사용 권장

    pass_mask = (
        (df["qc_blur"] >= blur_min) &
        (df["exp_lo"]  >= exp_lo_min) &
        (df["exp_hi"]  <= exp_hi_max) &
        (df["skin_cov"]>= skin_cov_min)
    )

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df_pass = df[pass_mask].copy()
    df_fail = df[~pass_mask].copy()
    df_pass.to_csv(out/"pass.csv", index=False)
    df_fail.to_csv(out/"fail.csv", index=False)

    # 선택: 파일 복사(정렬 이미지+3채널 cond 맵)
    if copy:
        cur_root = out/"curated"
        for _, row in df_pass.iterrows():
            rel = row["rel_path"]
            # aligned 이미지
            src_img = output_root / "aligned" / rel
            # 3채널 cond(.npy)
            src_cond = output_root / "maps" / "cond" / Path(rel).with_suffix(".npy")
            # 오버레이(선택)
            src_prev = output_root / "previews" / (Path(rel).stem + "_preview.png")

            for s, sub in [(src_img, "images"), (src_cond, "cond"), (src_prev, "previews")]:
                if s.exists():
                    dst = cur_root / sub / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(s, dst)

    print(f"[OK] pass={len(df_pass)}, fail={len(df_fail)}; wrote to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()
    main(args.config, args.labels, args.out, copy=args.copy)
