# src/metrics/pore.py
import cv2
import numpy as np

def _ensure_sigma_order(sigma_small: float, sigma_large: float):
    """sigma_large > sigma_small 보장. 같거나 역전이면 안전하게 교정."""
    s1 = float(sigma_small); s2 = float(sigma_large)
    if s2 <= s1:
        s1, s2 = min(s1, s2), max(s1, s2)
        if s2 <= s1:
            s2 = s1 + 0.1  # 최소 간격 확보
    return s1, s2

def dog_bandpass(gray, sigma_small=1.0, sigma_large=2.5, border=cv2.BORDER_REFLECT101):
    """
    Difference of Gaussians (DoG) 대역통과. 반사 보더 사용으로 가장자리 아티팩트 최소화.
    """
    sigma_small, sigma_large = _ensure_sigma_order(sigma_small, sigma_large)
    blur_s = cv2.GaussianBlur(gray, (0, 0), sigma_small, borderType=border)
    blur_l = cv2.GaussianBlur(gray, (0, 0), sigma_large, borderType=border)
    return blur_s - blur_l

def normalize_by_percentile(src, mask=None, p=(5, 95)):
    """
    마스크 내부 퍼센타일 기반 [0..1] 정규화. (로버스트, 조명/대비 변화에 견고)
    """
    src = src.astype(np.float32)
    if mask is not None:
        vals = src[mask > 0].ravel()
    else:
        vals = src.ravel()
    if vals.size < 10:
        dst = np.zeros_like(src, dtype=np.float32)
        if mask is not None:
            dst[mask == 0] = 0.0
        return dst
    lo, hi = np.percentile(vals, p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        dst = np.zeros_like(src, dtype=np.float32)
        if mask is not None:
            dst[mask == 0] = 0.0
        return dst
    denom = max(1e-6, hi - lo)
    dst = np.clip((src - lo) / denom, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        dst[mask == 0] = 0.0
    return dst

def pore_map_and_score(img_bgr,
                       skin_mask,
                       sigma_small=1.0,
                       sigma_large=2.5,
                       map_percentiles=(5, 95)):
    """
    반환:
      por_map01 : float32, [0..1], 피부 영역만 값
      score     : por_map01[skin] 평균 (연속값)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dog = np.abs(dog_bandpass(gray, sigma_small, sigma_large))  # 고주파 양의 반응
    por = normalize_by_percentile(dog, mask=skin_mask, p=map_percentiles)
    score = float(por[skin_mask > 0].mean()) if (skin_mask > 0).any() else 0.0
    return por.astype(np.float32), score

def pore_score(img_bgr,
               skin_mask,
               sigma_small=1.0,
               sigma_large=2.5,
               map_percentiles=(5, 95)):
    """
    스칼라 점수도 맵과 동일한 경로를 사용해 0..1 연속값 보장.
    """
    _, s = pore_map_and_score(
        img_bgr, skin_mask,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        map_percentiles=map_percentiles
    )
    return float(s)
