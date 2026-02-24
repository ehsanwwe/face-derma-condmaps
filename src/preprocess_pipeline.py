import argparse
import yaml
import cv2
import numpy as np
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# 현재 파일 위치를 기준으로 src 폴더를 path에 추가하여 모듈 임포트 지원
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent  # src/.. -> project root
sys.path.append(str(project_root)) # 프로젝트 루트 추가 (src.detectors... 등 사용 가능)

# 모듈 임포트 (src.xxx 형식으로 접근)
try:
    from src.detectors.insightface_detector import InsightfaceAligner
    from src.parsing.bisenet_parser import ZLLBiSeNetParser
    from src.metrics.redness import redness_map_and_score
    from src.metrics.wrinkle import wrinkle_map_and_score
    from src.metrics.pore import pore_map_and_score
    from src.qc.quality import blur_var_laplacian, exposure_percentiles
except ImportError as e:
    # src 폴더 내부에서 실행할 경우를 대비한 fallback
    sys.path.append(str(current_dir))
    from detectors.insightface_detector import InsightfaceAligner
    from parsing.bisenet_parser import ZLLBiSeNetParser
    from metrics.redness import redness_map_and_score
    from metrics.wrinkle import wrinkle_map_and_score
    from metrics.pore import pore_map_and_score
    from qc.quality import blur_var_laplacian, exposure_percentiles

def main(cfg):
    # --- 설정 로드 ---
    paths = cfg['paths']
    parsing_cfg = cfg['parsing']
    metrics_cfg = cfg['metrics']
    runtime_cfg = cfg['runtime']
    
    device = "cuda" if (runtime_cfg.get('use_cuda', True) and torch.cuda.is_available()) else "cpu"
    print(f"Running on {device}")

    # 경로 설정
    image_root = Path(paths['image_root'])
    output_root = Path(paths['output_root'])
    
    # 저장 경로 생성
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "maps").mkdir(parents=True, exist_ok=True)
    
    # 미리보기 폴더
    previews_root = output_root / "previews"
    if runtime_cfg.get('save_previews', False):
        previews_root.mkdir(parents=True, exist_ok=True)

    # 모델 초기화
    # 1. Aligner
    ctx_id = 0 if device == 'cuda' else -1
    aligner = InsightfaceAligner(ctx_id=ctx_id, det_size=(640, 640))
    align_size = int(cfg.get('align', {}).get('out_size', 512))

    # 2. Parser (BiSeNet)
    parser = ZLLBiSeNetParser(
        ckpt_path=paths['bisenet_ckpt'],
        input_size=512, # BiSeNet 입력은 512 고정 (가중치 호환성)
        device=device,
        skin_idx=1
    )

    # 이미지 리스트 로드
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = [p for p in image_root.rglob('*') if p.suffix.lower() in valid_ext]
    print(f"Found {len(image_paths)} images in {image_root}")

    # 파라미터 로드
    labels_cfg = parsing_cfg.get('labels', {})
    pp_cfg = parsing_cfg.get('postprocess', {})
    
    # 라벨 인덱스
    idx_skin = int(labels_cfg.get('skin', 1))
    idx_nose = int(labels_cfg.get('nose', 10))
    idx_ulip = int(labels_cfg.get('upper_lip', 12))
    idx_llip = int(labels_cfg.get('lower_lip', 13))
    idx_hair = int(labels_cfg.get('hair', 17))
    idx_eb_l = int(labels_cfg.get('eyebrow_left', 2))
    idx_eb_r = int(labels_cfg.get('eyebrow_right', 3))

    # --- 처리 루프 ---
    for p in tqdm(image_paths, desc="Preprocessing"):
        try:
            filename = p.name
            file_stem = p.stem
            
            # 1. 이미지 로드
            bgr = cv2.imread(str(p))
            if bgr is None: continue

            # 2. 얼굴 감지 및 정렬
            faces = aligner.detect(bgr)
            face = aligner.biggest_face(faces)
            if face is None: continue
            
            aligned = aligner.crop_aligned(bgr, face, out_size=align_size)
            if aligned is None: continue

            # 3. 마스크 파싱 (BiSeNet)
            # BiSeNetParser 내부에서 512 리사이즈 및 ImageNet 정규화 처리함 -> "이상한 마스크" 문제 해결!
            parsing_map = parser.parse(aligned) # (H, W) int16
            
            # 4. 마스크 후처리
            skin_mask = (parsing_map == idx_skin).astype(np.uint8)
            
            # 코 포함
            if pp_cfg.get('include_nose', True):
                nose_mask = (parsing_map == idx_nose).astype(np.uint8)
                skin_mask = np.clip(skin_mask + nose_mask, 0, 1)

            # 입술 제외
            if pp_cfg.get('exclude_lips', True):
                lip_mask = ((parsing_map == idx_ulip) | (parsing_map == idx_llip)).astype(np.uint8)
                skin_mask[lip_mask > 0] = 0
            
            # 눈썹 제외
            if pp_cfg.get('exclude_eyebrows', True):
                brow_mask = ((parsing_map == idx_eb_l) | (parsing_map == idx_eb_r)).astype(np.uint8)
                skin_mask[brow_mask > 0] = 0

            # 헤어 경계 정리
            hair_bound = int(pp_cfg.get('exclude_hair_boundary_px', 0))
            if hair_bound > 0:
                hair_mask = (parsing_map == idx_hair).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*hair_bound+1, 2*hair_bound+1))
                hair_dilated = cv2.dilate(hair_mask, kernel, iterations=1)
                skin_mask[hair_dilated > 0] = 0

            # 전체 마스크 침식
            erode_px = int(pp_cfg.get('erode_skin_px', 0))
            if erode_px > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
                skin_mask = cv2.erode(skin_mask, kernel, iterations=1)

            # 스무딩
            open_k = int(pp_cfg.get('smooth_open', 0))
            if open_k > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_k+1, 2*open_k+1))
                skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k)
            
            close_k = int(pp_cfg.get('smooth_close', 0))
            if close_k > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_k+1, 2*close_k+1))
                skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k)

            # 유효 마스크 확인 (너무 작으면 스킵)
            if np.sum(skin_mask) < (align_size * align_size * 0.05): continue

            # 5. 특징 추출
            # Redness
            red_cfg = metrics_cfg['redness']
            red_map, _ = redness_map_and_score(
                aligned, skin_mask,
                a_clip=tuple(red_cfg.get('a_star_clip_percentiles', [5, 95])),
                erode_px=int(red_cfg.get('cheek_mask_erode', 3)),
                illum_cfg=red_cfg.get('l_guided', None)
            )
            
            # Wrinkle
            wrk_cfg = metrics_cfg['wrinkle']
            wrk_map, _ = wrinkle_map_and_score(
                aligned, skin_mask,
                ksize=int(wrk_cfg.get('laplacian_ksize', 3)),
                gabor_cfg=wrk_cfg.get('gabor', {}),
                map_percentiles=tuple(wrk_cfg.get('map_percentiles', [5, 95]))
            )
            
            # Pore
            por_cfg = metrics_cfg['pore']
            por_map, _ = pore_map_and_score(
                aligned, skin_mask,
                sigma_small=float(por_cfg['dog'].get('sigma_small', 1.0)),
                sigma_large=float(por_cfg['dog'].get('sigma_large', 2.5)),
                map_percentiles=tuple(por_cfg.get('map_percentiles', [5, 95]))
            )

            # 6. 저장
            # 이미지
            cv2.imwrite(str(output_root / "images" / f"{file_stem}.png"), aligned)
            
            # .npy (4채널: Red, Wrinkle, Pore, Mask)
            maps_stack = np.stack([red_map, wrk_map, por_map, skin_mask], axis=0).astype(np.float32)
            np.save(str(output_root / "maps" / f"{file_stem}_cond.npy"), maps_stack)
            
            # 미리보기
            if runtime_cfg.get('save_previews', False):
                # 텍스처 맵 컬러화
                def to_color(m):
                    return cv2.applyColorMap((np.clip(m,0,1)*255).astype(np.uint8), cv2.COLORMAP_JET)
                
                vis = cv2.merge([
                    (por_map * 255).astype(np.uint8),
                    (wrk_map * 255).astype(np.uint8),
                    (red_map * 255).astype(np.uint8)
                ])
                vis[skin_mask == 0] = 0
                
                mask_vis = cv2.cvtColor((skin_mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                concat = np.hstack([aligned, mask_vis, vis])
                cv2.imwrite(str(previews_root / f"{file_stem}_prev.png"), concat)

        except Exception as e:
            print(f"Failed {p.name}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    main(cfg)