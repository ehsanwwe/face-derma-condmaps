import cv2
import numpy as np
from insightface.app import FaceAnalysis

class InsightfaceAligner:
    def __init__(self, ctx_id=0, det_size=(640, 640)):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, img_bgr):
        return self.app.get(img_bgr)

    def biggest_face(self, faces):
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    def crop_aligned(self, img_bgr, face, out_size=512):
        # 5점 랜드마크 기반 정렬
        kps = face.kps  # (5,2): left_eye, right_eye, nose, left_mouth, right_mouth
        dst = np.array([[0.30,0.35],
                        [0.70,0.35],
                        [0.50,0.55],
                        [0.35,0.75],
                        [0.65,0.75]], dtype=np.float32) * out_size
        src = kps.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            # fallback: bbox crop
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = img_bgr.shape[:2]
            x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                return None
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
        aligned = cv2.warpAffine(img_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR)
        return aligned