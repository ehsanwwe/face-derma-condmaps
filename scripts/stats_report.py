"""
점검 리포트 생성 (분포/상관/이상치 제안)

실행 명령어:
conda activate face-parsing
python scripts\stats_report.py --labels D:\face_parsing\outputs\labels.csv --out D:\face_parsing\outputs\reports
"""

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def main(labels_csv, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(labels_csv)

    # 기본 컬럼 확인
    needed = ["rel_path","wrinkle","pore","redness","qc_blur","exp_lo","exp_hi","skin_cov"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in labels.csv: {missing}")

    # 숫자 캐스팅
    for c in needed[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()

    # 히스토그램들
    fig = plt.figure(figsize=(12,8))
    for i, c in enumerate(["wrinkle","pore","redness","qc_blur","exp_lo","exp_hi","skin_cov"]):
        ax = plt.subplot(3,3,i+1)
        ax.hist(df[c].values, bins=50)
        ax.set_title(c)
    plt.tight_layout()
    fig.savefig(out/"hists.png", dpi=150)
    plt.close(fig)

    # 상관행렬
    corr_cols = ["wrinkle","pore","redness","qc_blur","skin_cov"]
    C = df[corr_cols].corr()
    fig = plt.figure(figsize=(5,4))
    im = plt.imshow(C, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right')
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation")
    plt.tight_layout()
    fig.savefig(out/"corr.png", dpi=150)
    plt.close(fig)

    # 이상치 제안 (경험적+분위수 기반)
    sugg = {}
    # 흐림: 너무 낮으면 fail
    sugg["qc_blur_min"] = float(max(40.0, df["qc_blur"].quantile(0.05)))
    # 노출: 하단/상단 안전범위
    sugg["exp_lo_min"]  = float(max(1.0, df["exp_lo"].quantile(0.05)))
    sugg["exp_hi_max"]  = float(min(250.0, df["exp_hi"].quantile(0.95)))
    # 마스크 커버리지: 최소 0.25~0.30 권장
    sugg["skin_cov_min"]= float(max(0.30, df["skin_cov"].quantile(0.05)))

    # 점수 기반(선택): 극단값 제거를 위한 IQR
    for k in ["wrinkle","pore","redness"]:
        q1, q3 = df[k].quantile([0.25,0.75])
        iqr = q3 - q1
        lo  = max(0.0, q1 - 1.5*iqr)
        hi  = min(1.0, q3 + 1.5*iqr)
        sugg[f"{k}_range"] = [float(lo), float(hi)]

    with open(out/"suggested_qc.yaml","w",encoding="utf-8") as f:
        yaml.safe_dump(sugg, f, sort_keys=False, allow_unicode=True)

    # 후보 이상치 목록 저장
    mask_fail = (
        (df["qc_blur"] < sugg["qc_blur_min"]) |
        (df["exp_lo"]  < sugg["exp_lo_min"])  |
        (df["exp_hi"]  > sugg["exp_hi_max"])  |
        (df["skin_cov"]< sugg["skin_cov_min"])
    )
    outliers = df[mask_fail]
    outliers.to_csv(out/"qc_candidates.csv", index=False)
    print(f"[OK] wrote {out/'hists.png'}, {out/'corr.png'}, {out/'suggested_qc.yaml'}, {out/'qc_candidates.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.labels, args.out)
