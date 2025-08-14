import cv2
import numpy as np

def blur_var_laplacian(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def exposure_percentiles(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(g, [1, 99])
    return float(lo), float(hi)
