import os
import warnings

# Suppress CUDA warnings since we are using CPU
warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider' is not in available provider names")

# Ensure only CPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Your imports and code below...
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ===== ماژول‌های پروژه ===== #
from detectors.insightface_detector import InsightfaceAligner
from src.parsing.bisenet_parser import ZLLBiSeNetParser
from src.utils.viz import make_cond_tile
from src.metrics.redness import redness_map_and_score
from src.metrics.wrinkle import wrinkle_map_and_score
from src.metrics.pore import pore_map_and_score


# ===== endpoint ===== #
import base64
import traceback
import uuid

# ===== FastAPI ===== #
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== آماده‌سازی مدل‌ها ===== #
aligner = InsightfaceAligner(ctx_id=-1, det_size=(640, 640))
parser = ZLLBiSeNetParser(
    ckpt_path="models/79999_iter.pth",  # مسیر وزن BiSeNet رو اینجا بذار
    input_size=512,
    device="cpu",  # Explicitly set to "cpu"
    skin_idx=1,
    debug_hist=False
)




@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        # خواندن تصویر آپلود شده
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return JSONResponse({"error": "invalid image"}, status_code=400)

        # تشخیص چهره و align
        faces = aligner.detect(bgr)
        face = aligner.biggest_face(faces)
        if face is None:
            return JSONResponse({"error": "no face detected"}, status_code=400)

        # اگر در پروژه‌ات متغیر align_size تعریف است از آن استفاده کن؛ در غیر اینصورت 512
        aligned = aligner.crop_aligned(bgr, face, out_size=512)
        if aligned is None:
            return JSONResponse({"error": "align failed"}, status_code=500)

        # پارسینگ → ماسک پوست (اگر parser نداریم ماسکِ تمام‌یک‌ها)
        if 'parser' in globals() and parser is not None:
            parsing_map = parser.parse(aligned)
            skin_mask = parser.skin_mask(parsing_map)
            if skin_mask is None:
                # از ماسک یک‌ها استفاده کن تا None نباشد
                h, w = aligned.shape[:2]
                skin_mask = np.ones((h, w), dtype=np.uint8)
        else:
            h, w = aligned.shape[:2]
            skin_mask = np.ones((h, w), dtype=np.uint8)

        # محاسبه‌ی مپ‌ها و اسکورها (بازگشتی: map, score)
        red_map, red_score = redness_map_and_score(aligned, skin_mask)
        wrk_map, wrk_score = wrinkle_map_and_score(aligned, skin_mask)
        por_map, por_score = pore_map_and_score(aligned, skin_mask)

        # تولید tile و ذخیره در فایل منحصربه‌فرد
        out_dir = Path("static")
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"tile_{uuid.uuid4().hex}.png"
        out_path = out_dir / fname

        # make_cond_tileِ شما فایل را ذخیره و مسیر را برمی‌گرداند (همان نسخه‌ی قبلی)
        saved = make_cond_tile(aligned, red_map, wrk_map, por_map, save_path=str(out_path), alpha=0.45)

        # بعضی نسخه‌ها مسیر برگشتی می‌دهند یا None؛ اطمینان از مسیر نهایی
        if saved is None:
            saved_path = out_path
        else:
            saved_path = Path(saved)

        # خواندن فایل و تبدیل به base64
        with open(saved_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        # پاک کردن فایل موقت (اختیاری؛ اگر می‌خواهی نگه داری، این بخش را حذف کن)
        try:
            saved_path.unlink()
        except Exception:
            pass

        # پاسخ JSON شامل base64 و اسکورها
        return {
            "image_base64": img_base64,
            "red_score": float(red_score),
            "wrinkle_score": float(wrk_score),
            "pore_score": float(por_score)
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ===== اجرای مستقیم ===== #
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=80, reload=True,loop="asyncio")