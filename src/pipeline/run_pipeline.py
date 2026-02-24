"""
실행 명령어:
python -m src.pipeline.run_pipeline --config config/config.yaml
"""

# ===== library ===== #
import argparse, yaml, csv
import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from detectors.insightface_detector import InsightfaceAligner

# ===== utils ===== #
from src.utils.io import list_images
from src.utils.viz import make_cond_tile

# ===== parsing ===== #
from src.parsing.bisenet_parser import ZLLBiSeNetParser

# ===== metrics ===== #
from src.metrics.redness import redness_score, redness_map_and_score
from src.metrics.wrinkle import wrinkle_score, wrinkle_map_and_score
from src.metrics.pore import pore_score, pore_map_and_score

# ===== qc ===== #
from src.qc.quality import blur_var_laplacian, exposure_percentiles



# ===== main ===== #
def main(cfg):

    # root path / parsing
    paths         = cfg['paths'];   runtime = cfg['runtime']
    parsing       = cfg['parsing']; metrics = cfg['metrics']

    # align 사이즈 config에서 읽기
    align_size    = int(cfg.get('align', {}).get('out_size', 512))
    save_aligned  = bool(cfg.get('runtime', {}).get('save_aligned', True))

    # paths
    image_root    = Path(paths['image_root'])
    out_root      = Path(paths['output_root']); out_root.mkdir(parents=True, exist_ok=True)
    aligned_root  = out_root / "aligned"
    fail_log_path = out_root / "failures.csv"

    labels_csv    = out_root / "labels.csv"

    previews_root = out_root / "previews"
    if runtime.get('save_previews', True):
        previews_root.mkdir(parents=True, exist_ok=True)

    # 이미지 경로
    image_paths   = list(list_images(image_root))
    image_paths.sort()

    # detector / aligner
    ctx_id        = 0 if runtime.get('use_cuda', True) else -1
    aligner       = InsightfaceAligner(ctx_id=ctx_id, det_size=(640, 640))

    # 출력 경로
    if save_aligned:
        aligned_root.mkdir(parents=True, exist_ok=True)

    # 실패 로그
    fail_f = open(fail_log_path, 'w', newline='', encoding='utf-8')
    fail_w = csv.writer(fail_f); fail_w.writerow(["rel_path","reason"])

    # 라벨 CSV 헤더에 skin_cov 추가
    # with open(labels_csv, 'w', newline='', encoding='utf-8') as f:
    #     csv_w = csv.writer(f)
    #     csv_w.writerow(["rel_path","wrinkle","pore","redness","qc_blur","exp_lo","exp_hi","skin_cov"])

    # parser (BiSeNet)
    backend = parsing.get('backend', 'zll_bisenet').lower()
    parser  = None
    if backend == 'zll_bisenet':
        parser = ZLLBiSeNetParser(
            ckpt_path=paths.get('bisenet_ckpt',''),
            input_size=int(parsing.get('input_size',512)),
            device="cuda" if runtime.get('use_cuda', True) else "cpu",
            skin_idx=int(parsing.get('labels',{}).get('skin',1)),
            debug_hist=False
        )

    # 라벨 파라미터
    labels_cfg    = parsing.get('labels', {})
    skin_idx      = int(labels_cfg.get('skin', 1))
    nose_idx      = int(labels_cfg.get('nose', 10))
    ul_idx        = int(labels_cfg.get('upper_lip', 12))
    ll_idx        = int(labels_cfg.get('lower_lip', 13))
    hair_idx      = int(labels_cfg.get('hair', 17))
    ebl_idx       = int(labels_cfg.get('eyebrow_left', 2))
    ebr_idx       = int(labels_cfg.get('eyebrow_right', 3))

    # 후처리 파라미터
    pp            = parsing.get('postprocess', {})
    include_nose  = bool(pp.get('include_nose', True))
    open_k        = int(pp.get('smooth_open', 0))
    close_k       = int(pp.get('smooth_close', 0))
    erode_px      = int(pp.get('erode_skin_px', 0))
    excl_lips     = bool(pp.get('exclude_lips', True))
    excl_eb       = bool(pp.get('exclude_eyebrows', True))
    hair_bound_px = int(pp.get('exclude_hair_boundary_px', 0))

    # L_Guided
    red_cfg       = metrics['redness']
    illum_cfg     = red_cfg.get('l_guided', None)

    # 저장 파라미터
    save_maps     = bool(runtime.get('save_maps', True))
    save_png      = bool(runtime.get('save_maps_png', True))
    save_npy      = bool(runtime.get('save_maps_npy', True))
    save_tile     = bool(runtime.get('save_cond_tile_png', True))

    # 미리보기
    preview_count = 0
    max_prev = int(runtime.get('preview_max', 200))



    with open(labels_csv, 'w', newline='', encoding='utf-8') as f:
        csv_w = csv.writer(f)
        csv_w.writerow(["rel_path","wrinkle","pore","redness","qc_blur","exp_lo","exp_hi","skin_cov"])

        for p in tqdm(image_paths, desc="Processing"):
            try:
                rel = str(p.relative_to(image_root))
                bgr = cv2.imread(str(p))
                if bgr is None: 
                    fail_w.writerow([rel, "imread_failed"]); continue

                faces = aligner.detect(bgr)
                face  = aligner.biggest_face(faces)
                if face is None: 
                    fail_w.writerow([rel, "no_face"]); continue

                aligned = aligner.crop_aligned(bgr, face, out_size=align_size)
                if aligned is None:
                    fail_w.writerow([rel, "align_failed"]); continue
                
                if save_aligned:
                    apath = aligned_root / rel
                    apath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(apath), aligned)
            except Exception as e:
                fail_w.writerow([str(p), f"exception:{type(e).__name__}"])
                continue



            # ===== parsing → skin mask ===== #
            if parser is not None:
                parsing_map = parser.parse(aligned)
                skin_mask   = parser.skin_mask(parsing_map)

                # ===== 코 포함 옵션 ===== #
                if include_nose:
                    nose_mask = (parsing_map == nose_idx).astype(np.uint8)
                    skin_mask = np.clip(skin_mask + nose_mask, 0, 1).astype(np.uint8)

                # ===== 마스크 후처리 시작 (추가) ===== #
                # 1) 입술 제외
                if excl_lips:
                    lip_mask  = ((parsing_map == ul_idx) | (parsing_map == ll_idx)).astype(np.uint8)
                    skin_mask[lip_mask > 0] = 0

                # 2) 헤어 경계 억제 (헤어 dilate 후 제거)
                if hair_bound_px > 0:
                    hair_mask = (parsing_map == hair_idx).astype(np.uint8)
                    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*hair_bound_px+1, 2*hair_bound_px+1))
                    hair_dil  = cv2.dilate(hair_mask, k, iterations=1)
                    skin_mask[hair_dil > 0] = 0

                # 3) 피부 마스크 전체 erode (헤어라인 들쑥 줄이기)
                if erode_px > 0:
                    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
                    skin_mask = cv2.erode(skin_mask, k, iterations=1)

                # 4) 스무딩 (open/close)
                if open_k > 0:
                    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_k+1, 2*open_k+1))
                    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k)
                if close_k > 0:
                    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_k+1, 2*close_k+1))
                    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k)

                # 5) 눈썹
                if excl_eb:
                    brow_mask = ((parsing_map == ebl_idx) | (parsing_map == ebr_idx)).astype(np.uint8)
                    skin_mask[brow_mask > 0] = 0
                # ===== 마스크 후처리 끝 ===== #



            else:
                h, wimg   = aligned.shape[:2]
                skin_mask = np.ones((h, wimg), dtype=np.uint8)



            # ===== 스칼라 스코어 ===== #
            # redness score
            red = redness_score(
                aligned, skin_mask,
                a_clip=tuple(metrics['redness'].get('a_star_clip_percentiles', [5,95])),
                erode=int(metrics['redness'].get('cheek_mask_erode', 3)),
                illum_cfg=illum_cfg
            )
            # wrinkle score
            wrk = wrinkle_score(
                aligned, skin_mask,
                ksize=int(metrics['wrinkle'].get('laplacian_ksize', 3)),
                gabor_cfg=metrics['wrinkle'].get('gabor', {})
            )
            # pore score
            por = pore_score(
                aligned, skin_mask,
                sigma_small=float(metrics['pore']['dog'].get('sigma_small', 1.0)),
                sigma_large=float(metrics['pore']['dog'].get('sigma_large', 2.5))
            )



            # ===== 히트맵 생성 ===== #
            # redness map
            red_map, red_s = redness_map_and_score(
                aligned, skin_mask,
                a_clip      =tuple(red_cfg.get('a_star_clip_percentiles', [5, 95])),
                smooth_sigma=float(red_cfg.get('map_smooth_sigma', 1.0)),
                erode_px    =int(red_cfg.get('cheek_mask_erode', 3)),
                illum_cfg   =illum_cfg
            )
            # wrinkle map
            wrk_map, wrk_s2 = wrinkle_map_and_score(
                aligned, skin_mask,
                ksize          =int(metrics['wrinkle'].get('laplacian_ksize', 3)),
                gabor_cfg      =metrics['wrinkle'].get('gabor', {}),
                map_percentiles=tuple(metrics['wrinkle'].get('map_percentiles', [5,95])),
            )
            # pore map
            por_map, por_s2 = pore_map_and_score(
                aligned, skin_mask,
                sigma_small    =float(metrics['pore']['dog'].get('sigma_small', 1.0)),
                sigma_large    =float(metrics['pore']['dog'].get('sigma_large', 2.5)),
                map_percentiles=tuple(metrics['pore'].get('map_percentiles', [5,95])),
            )



            # ===== 경계 falloff: 맵 저장 직전에 경계에서 값을 점차 낮추는 가중치 적용 ===== #
            falloff_px = int(pp.get('boundary_falloff_px', 0))
            if falloff_px > 0 and (skin_mask > 0).any():
                dist     = cv2.distanceTransform((skin_mask>0).astype(np.uint8), cv2.DIST_L2, 3)
                w_fall   = np.clip(dist / float(max(1, falloff_px)), 0.0, 1.0).astype(np.float32)
                red_map *= w_fall; wrk_map *= w_fall; por_map *= w_fall



            # 모공 맵 안정화(선택)
            por_map  = cv2.medianBlur((por_map*255).astype(np.uint8), 3).astype(np.float32)/255.0
            por_map *= (skin_mask > 0).astype(np.float32)  # ← 재마스크



            # QC
            blur   = blur_var_laplacian(aligned)
            lo, hi = exposure_percentiles(aligned)



            # skin_cov 계산(마스크 0/1 평균)
            skin_cov = float(skin_mask.mean())



            # write
            # 맵 생성 직후 이미 이 값들을 갖고 있음: red_s, wrk_s2, por_s2
            csv_w.writerow([rel,
                        f"{wrk_s2:.4f}", f"{por_s2:.4f}", f"{red_s:.4f}",
                        f"{blur:.2f}", f"{lo:.1f}", f"{hi:.1f}",
                        f"{skin_cov:.4f}"])



            # preview
            if runtime.get('save_previews', True) and preview_count < max_prev:
                prev           = aligned.copy()
                cnts, _        = cv2.findContours((skin_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(prev, cnts, -1, (0,255,0), 1)

                txt            = f"W:{wrk:.2f} P:{por:.2f} R:{red:.2f} BLR:{blur:.1f}"
                cv2.putText(prev, txt, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(prev, txt, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

                out_path       = previews_root / (Path(rel).stem + "_preview.png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), prev)

                preview_count += 1



            # 스코어는 기존 계산값(wrk, por, red)과 거의 유사해야 함 (다르면 퍼센타일/정규화 차이)
            # 원하면 여기서 wrk,por,red를 wrk_s2,por_s2,red_s로 대체해도 됨.
            overlay_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            # rel 이 '00012/xxx.png' 같은 구조라면, 동일한 하위 디렉토리로 저장됨
            if save_maps:
                save_map_png_and_npy(out_root, "maps/redness",  rel, red_map, save_png, save_npy, overlay_src=overlay_rgb)
                save_map_png_and_npy(out_root, "maps/wrinkle",  rel, wrk_map, save_png, save_npy, overlay_src=overlay_rgb)
                save_map_png_and_npy(out_root, "maps/pore",     rel, por_map, save_png, save_npy, overlay_src=overlay_rgb)

                # 조건지도 3채널 NPY
                if save_maps and save_npy:
                    cond     = np.stack([red_map, wrk_map, por_map, skin_mask.astype(np.float32)], axis=0)
                    out_base = (out_root / "maps/cond" / rel).with_suffix('')
                    out_base.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(out_base) + ".npy", cond)

                # 타일 프리뷰 PNG 저장
                if save_tile:
                    tile_path = out_root / "tiles" / rel
                    tile_path = tile_path.with_suffix(".png")
                    make_cond_tile(aligned, red_map, wrk_map, por_map, save_path=tile_path, alpha=0.45)



        fail_f.close()
        print(f"[OK] Failure log: {fail_log_path}")



    print(f"[OK] Wrote labels: {labels_csv}")
    if runtime.get('save_previews', True):
        print(f"[OK] Previews saved to: {previews_root} ({preview_count} items)")



def to_uint8(m01):
    m01 = np.clip(m01, 0.0, 1.0)
    return (m01 * 255.0).astype(np.uint8)



def save_map_png_and_npy(root_dir, subdir, rel, map01, save_png=True, save_npy=True, overlay_src=None):
    """
    root_dir/subdir/rel.png (또는 .npy)로 저장.
    overlay_src가 있으면 컬러맵을 덧씌운 오버레이 PNG도 함께 저장.
    """
    out_base = (Path(root_dir) / subdir / rel).with_suffix('')
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if save_npy:
        np.save(str(out_base) + ".npy", map01.astype(np.float32))

    if save_png:
        m8 = to_uint8(map01)
        color = cv2.applyColorMap(m8, cv2.COLORMAP_JET)
        if overlay_src is not None:
            src = overlay_src.copy()
            # 0.6*src + 0.4*heat
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            over = (0.6 * src.astype(np.float32) + 0.4 * color_rgb.astype(np.float32)).astype(np.uint8)
            cv2.imwrite(str(out_base) + "_overlay.png", cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_base) + ".png", color)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    main(cfg)