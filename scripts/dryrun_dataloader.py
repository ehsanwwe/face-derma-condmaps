"""
로더 검증 - 학습 직전 빠른 테스트

실행 명령어:
python scripts\dryrun_dataloader.py --manifest D:\face_parsing\outputs\splits\manifest.jsonl --split train
"""

import argparse, json
import numpy as np
import cv2, torch
from torch.utils.data import Dataset, DataLoader

class FaceCondDataset(Dataset):
    def __init__(self, manifest_jsonl, split="train", size=512):
        self.size = size
        self.items = []
        with open(manifest_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("split") == split:
                    self.items.append(rec)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = cv2.imread(rec["image"])
        if img is None:
            raise FileNotFoundError(rec["image"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cond = np.load(rec["cond"]).astype(np.float32)  # [3,H,W], 0..1

        # 안전 리사이즈(필요 시)
        H, W = img.shape[:2]
        if (H, W) != (self.size, self.size):
            img  = cv2.resize(img,  (self.size, self.size), interpolation=cv2.INTER_AREA)
            cond = np.stack([cv2.resize(cond[i], (self.size,self.size), interpolation=cv2.INTER_AREA)
                             for i in range(3)], axis=0)

        # to tensor [C,H,W], 0..1
        img  = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
        cond = torch.from_numpy(cond).float()
        return img, cond

def main(manifest, split):
    ds = FaceCondDataset(manifest, split=split, size=512)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    for i, (x, c) in enumerate(dl):
        print(f"batch {i}: x={tuple(x.shape)} cond={tuple(c.shape)} x.minmax=({x.min():.3f},{x.max():.3f}) c.minmax=({c.min():.3f},{c.max():.3f})")
        if i == 2: break
    print("[OK] dataloader dry-run passed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    main(args.manifest, args.split)
