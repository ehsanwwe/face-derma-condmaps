# face-derma-condmaps

[![python](https://img.shields.io/badge/python-3.10%7C3.11-blue)]()
[![pytorch](https://img.shields.io/badge/pytorch-nightly%20CUDA12x-orange)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()

**얼굴 데이터셋**에서 얼굴을 **검출·정렬 → 파싱(BiSeNet)** 한 뒤, 피부 영역에 대해  
**주름(Wrinkle) / 모공(Pore) / 홍조(Redness)** 의 **3채널 조건지도(heatmaps + .npy)**와  
품질지표(QC), 미리보기, 학습용 매니페스트까지 한 번에 생성하는 파이프라인.  
후속 **Conditional GAN**(노화/회춘 등)의 입력 조건으로 바로 사용할 수 있도록 설계.

> **R/G/B = Redness / Wrinkle / Pore**  
> 출력: 개별 히트맵 PNG/Overlay, 원본 float32 NPY, 3채널 조건지도 `cond.npy`, `labels.csv`, QC/리포트/분할 스크립트

---

## 용도

- **데이터 준비 자동화**: 얼굴 정렬·파싱·후처리·특성 맵 산출까지 일괄 처리  
- **GAN 친화적 구조**: 3채널 조건지도(.npy) 및 `manifest.jsonl` 제공 → 별도 GAN 레포에서 즉시 사용  
- **품질관리(QC)**: 흐림/노출/피부 커버 등 기본 지표와 필터링 스크립트 포함

---

## 환경 및 요구사항

- **OS**: Windows 11 (tested)  
- **GPU**: NVIDIA RTX 50 시리즈 권장  
  - ⚠️ RTX 5080/5090 등 최신 카드의 경우 **PyTorch nightly + CUDA 12.x** 권장
- **Python**: 3.10 ~ 3.11 (conda 권장)

### 설치 (권장 절차)

```bash
# 새 가상환경
conda create -n face-parsing python=3.10 -y
conda activate face-parsing

# PyTorch (nightly + CUDA 12.x)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 필수 패키지
pip install opencv-python insightface onnxruntime-gpu pyyaml tqdm matplotlib pandas
```

---

## Docker (API server, Linux / CPU)

`server.py` is a FastAPI service (`POST /process`, image in -> 3-channel condition tile + scores out).

```bash
git clone --recurse-submodules <your-repo-url>
cd face-derma-condmaps
# put the BiSeNet weights in models/79999_iter.pth  (see models/README.md)
docker compose up -d --build
```

- Host port is **6464** by default (container listens on 8000). Change it with `API_PORT=7000 docker compose up -d`.
- Swagger UI: http://localhost:6464/docs
- Test: `curl -F "file=@face.jpg" http://localhost:6464/process`
- The image is CPU-only; InsightFace `buffalo_l` and the resnet18 backbone weights are baked in at build time. `models/` is mounted read-only.

---

## 클론 가이드

```bash
git clone https://github.com/Keddmon/face-derma-condmaps.git
cd face-derma-condmaps
git submodule update --init --recursive
```

---

## 라이선스/출처

- **레포**: MIT License
- **사용 모델/코드**:
    - InsightFace (MIT)
    - BiSeNet (CelebA-HQ 파생, 가중치 `79999_iter.pth`는 사용자 소지 파일)
- **데이터**: FFHQ(NVLabs) - 데이터 라이선스/정책 준수

---
