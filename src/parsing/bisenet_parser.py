import os, sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

class ZLLBiSeNetParser:
    """
    zllrunning/face-parsing.PyTorch의 BiSeNet을 import하여
    79999_iter.pth(celebaMask-HQ 기반) 체크포인트로 추론.
    """
    def __init__(self, ckpt_path:str, input_size:int=512, device:str="cuda", skin_idx:int=1, debug_hist:bool=False):
        self.device = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
        self.input_size = input_size
        self.skin_idx = int(skin_idx)
        self.debug_hist = debug_hist

        # third_party에 있는 공식 레포 import
        root = Path(__file__).resolve().parents[2]  # 프로젝트 루트
        third_party = root / "third_party" / "face-parsing.PyTorch"
        model_py = third_party / "model.py"
        if not model_py.exists():
            raise FileNotFoundError(
                f"face-parsing.PyTorch not found at {third_party}. "
                f"Run: git clone https://github.com/zllrunning/face-parsing.PyTorch third_party/face-parsing.PyTorch"
            )
        sys.path.append(str(third_party))
        from model import BiSeNet  # type: ignore

        n_classes = 19  # CelebAMask-HQ의 일반적 클래스 수
        self.net = BiSeNet(n_classes=n_classes).to(self.device).eval()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"BiSeNet checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        sd = state.get('state_dict', state)
        self.net.load_state_dict(sd, strict=False)

        # ImageNet 정규화
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        img = cv2.resize(img_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2,0,1))[None]  # (1,3,H,W)
        return img, (h, w)

    def parse(self, img_bgr):
        """
        입력: BGR uint8 (H,W,3)
        출력: parsing_map (H,W) int16 [0..18]
        """
        x, (h, w) = self._preprocess(img_bgr)
        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            out = self.net(x)[0]  # (B, n_classes, H, W)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.int16)

        if self.debug_hist:
            uniq, cnt = np.unique(parsing, return_counts=True)
            print("[BiSeNet] label histogram:", dict(zip(uniq.tolist(), cnt.tolist())))
        return parsing

    def skin_mask(self, parsing_map):
        return (parsing_map == self.skin_idx).astype(np.uint8)
