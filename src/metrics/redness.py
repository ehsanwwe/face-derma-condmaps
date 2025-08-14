# src/metrics/redness.py
import cv2
import numpy as np

def _erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if not erode_px or erode_px <= 0:
        return mask.astype(np.uint8)
    k = 2 * int(erode_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

def _normalize01(x, mask=None, p=(1, 99)):
    x = x.astype(np.float32)
    vals = x[mask > 0].ravel() if mask is not None else x.ravel()
    if vals.size < 10:
        return np.zeros_like(x, np.float32)
    lo, hi = np.percentile(vals, p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, np.float32)
    y = (x - lo) / max(1e-6, hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return y

def _l_base(L, method="guided", radius=15, eps=1e-3):
    """밝기 L을 평탄화한 base(조명 성분) 추정. ximgproc 없으면 자동 폴백."""
    L = L.astype(np.float32)
    method = (method or "guided").lower()
    if method == "guided":
        # 필요: opencv-contrib-python
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            gf = cv2.ximgproc.guidedFilter(guide=L, src=L, radius=int(radius), eps=float(eps))
            return gf
        # 폴백
        method = "bilateral"
    if method == "bilateral":
        # 공간/색 표준편차를 radius, eps에 연동
        d = max(5, int(2 * radius + 1))
        sigma_color = max(10.0, 50.0 * float(eps)**-0.5)  # 완만한 경험치
        sigma_space = max(5.0, float(radius))
        return cv2.bilateralFilter(L, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    # 최종 폴백: Gaussian
    return cv2.GaussianBlur(L, (0, 0), max(1.0, float(radius) / 2.0))

def redness_map_and_score(img_bgr,
                          skin_mask,
                          a_clip=(5, 95),
                          smooth_sigma=1.0,
                          erode_px: int = 0,
                          illum_cfg: dict | None = None):
    """
    반환:
      red_map01 : [0..1] float32, 피부 영역만 값
      score     : red_map01[skin].mean()
    illum_cfg (선택):
      { use: bool, method: guided|bilateral|gaussian, radius: int, eps: float, alpha: float, clip_w: [lo, hi] }
      - 밝은 영역(하이라이트) 과대평가를 줄이기 위해 L-base로 가중치 w를 만들어 redness에 곱함.
    """
    mask = _erode_mask(skin_mask, erode_px)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L  = lab[:, :, 0]     # 0..255
    a  = lab[:, :, 1]     # 0..255, 128이 중립

    # a*의 양수 성분(적색)만
    pos = np.maximum(a - 128.0, 0.0).astype(np.float32)

    # --- L-guided 보정 (옵션) ---
    if illum_cfg and illum_cfg.get("use", False):
        method = illum_cfg.get("method", "guided")
        radius = int(illum_cfg.get("radius", 15))
        eps    = float(illum_cfg.get("eps", 1e-3))
        alpha  = float(illum_cfg.get("alpha", 1.0))
        clip_w = illum_cfg.get("clip_w", [0.5, 1.5])
        w_lo, w_hi = float(clip_w[0]), float(clip_w[1])

        Lb   = _l_base(L, method=method, radius=radius, eps=eps)
        Lb01 = _normalize01(Lb, mask=mask, p=(1, 99)) + 1e-6
        m    = float(Lb01[mask > 0].mean()) if (mask > 0).any() else 0.5
        w    = np.power(np.clip(m / Lb01, w_lo, w_hi), alpha).astype(np.float32)
        pos  = pos * w
    # -----------------------------

    vals = pos[mask > 0]
    if vals.size < 10:
        return np.zeros(a.shape, np.float32), 0.0

    # 로버스트 정규화 (퍼센타일, 마스크 내부 기준)
    p_lo, p_hi = np.percentile(vals, a_clip)
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        return np.zeros(a.shape, np.float32), 0.0

    denom = max(1e-6, p_hi - p_lo)
    red = np.zeros_like(pos, dtype=np.float32)
    red_in = (pos[mask > 0] - p_lo) / denom
    red[mask > 0] = np.clip(red_in, 0.0, 1.0)

    if smooth_sigma and smooth_sigma > 0:
        red = cv2.GaussianBlur(red, (0, 0), float(smooth_sigma))

    score = float(red[mask > 0].mean()) if (mask > 0).any() else 0.0
    return red, score

def redness_score(img_bgr,
                  skin_mask,
                  a_clip=(5, 95),
                  erode=3,
                  illum_cfg: dict | None = None):
    # 스칼라도 동일 경로 사용 (일관성)
    _, score = redness_map_and_score(
        img_bgr, skin_mask,
        a_clip=a_clip,
        smooth_sigma=0.0,
        erode_px=int(erode) if erode is not None else 0,
        illum_cfg=illum_cfg
    )
    return float(score)
