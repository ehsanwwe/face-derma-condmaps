# src/metrics/wrinkle.py
import cv2
import numpy as np

def _ensure_odd(n: int) -> int:
    # 라플라시안/가보 커널 크기는 홀수여야 함
    return n if (n % 2 == 1) else (n + 1)

def laplacian_map(gray, ksize=3):
    ksize = max(1, _ensure_odd(int(ksize)))
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return np.abs(lap)

def gabor_bank_map(gray, thetas=(0, 45, 90, 135), lambdas=(4, 8), sigmas=(2.0, 3.0)):
    """
    다양한 방향/주기의 가보 반응 평균.
    커널 크기는 sigma/lambda에 맞춰 동적으로 설정(경계 artifact 완화용 반사 보더).
    """
    acc = None; cnt = 0
    for th in thetas:
        theta = th * np.pi / 180.0
        for lam in lambdas:
            for sigma in sigmas:
                # 커널 크기: 대략 6*sigma 범위가 충분, 홀수로 보정
                ksz = _ensure_odd(int(np.round(6 * float(sigma))))
                ksz = max(ksz, 7)  # 너무 작지 않게
                kern = cv2.getGaborKernel((ksz, ksz), sigma, theta, float(lam), gamma=0.5, psi=0)
                resp = cv2.filter2D(gray, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT101)
                resp = np.abs(resp)
                acc = resp if acc is None else (acc + resp)
                cnt += 1
    if cnt == 0:
        return np.zeros_like(gray, dtype=np.float32)
    return acc / float(cnt)

def normalize_by_percentile(src, mask=None, p=(5, 95)):
    """
    src를 (p_lo, p_hi) 로버스트 퍼센타일로 [0..1] 정규화.
    mask가 있으면 마스크 내부 픽셀만 통계 계산.
    """
    dst = np.zeros_like(src, dtype=np.float32)
    if mask is None:
        vals = src.reshape(-1)
    else:
        vals = src[mask > 0].reshape(-1)
    if vals.size < 10:
        return dst
    lo, hi = np.percentile(vals, p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return dst
    denom = max(1e-6, hi - lo)
    dst = np.clip((src - lo) / denom, 0.0, 1.0)
    if mask is not None:
        dst[mask == 0] = 0.0
    return dst

def wrinkle_map_and_score(img_bgr, skin_mask, ksize=3, gabor_cfg=None, map_percentiles=(5, 95)):
    """
    주름 맵(wrk, [0..1])과 연속 점수(mean of wrk inside mask) 반환.
    """
    if gabor_cfg is None:
        gabor_cfg = {}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap = laplacian_map(gray, ksize=ksize)
    gab = gabor_bank_map(
        gray,
        thetas=gabor_cfg.get("thetas", [0, 45, 90, 135]),
        lambdas=gabor_cfg.get("lambdas", [4, 8]),
        sigmas=gabor_cfg.get("sigmas", [2.0, 3.0]),
    )

    # 퍼센타일 기반 정규화(마스크 내부 기준)
    lap_n = normalize_by_percentile(lap, mask=skin_mask, p=map_percentiles)
    gab_n = normalize_by_percentile(gab, mask=skin_mask, p=map_percentiles)

    # 가중 결합(경험적 가중치)
    wrk = 0.6 * lap_n + 0.4 * gab_n
    wrk = np.clip(wrk, 0.0, 1.0).astype(np.float32)

    score = float(wrk[skin_mask > 0].mean()) if (skin_mask > 0).any() else 0.0
    return wrk, score

# ---- 호환용 스칼라 함수들(에너지 기반). 필요 시 유지 ----
def laplacian_energy(gray, ksize=3, mask=None):
    ksize = max(1, _ensure_odd(int(ksize)))
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    vals = lap[mask > 0].reshape(-1) if mask is not None else lap.reshape(-1)
    if vals.size < 10:
        return 0.0
    return float(np.mean(np.abs(vals)))

def gabor_bank_energy(gray, thetas=(0, 45, 90, 135), lambdas=(4, 8), sigmas=(2.0, 3.0), mask=None):
    acc = 0.0; cnt = 0
    for th in thetas:
        theta = th * np.pi / 180.0
        for lam in lambdas:
            for sigma in sigmas:
                ksz = _ensure_odd(int(np.round(6 * float(sigma))))
                ksz = max(ksz, 7)
                kern = cv2.getGaborKernel((ksz, ksz), sigma, theta, float(lam), gamma=0.5, psi=0)
                resp = cv2.filter2D(gray, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT101)
                vals = resp[mask > 0].reshape(-1) if mask is not None else resp.reshape(-1)
                if vals.size < 10:
                    continue
                acc += float(np.mean(np.abs(vals))); cnt += 1
    if cnt == 0:
        return 0.0
    return acc / cnt

def wrinkle_score(img_bgr, skin_mask, ksize=3, gabor_cfg=None, map_percentiles=(5, 95)):
    """
    스칼라 점수도 맵과 동일한 정규화 경로를 사용하여 0..1 연속값을 보장.
    → CSV의 wrinkle 컬럼이 1로 고정되는 문제 방지.
    """
    wrk_map, score = wrinkle_map_and_score(
        img_bgr, skin_mask,
        ksize=ksize,
        gabor_cfg=gabor_cfg,
        map_percentiles=map_percentiles
    )
    return float(score)
