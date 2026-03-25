# face-derma-condmaps 실행 및 활용 가이드
> 본 가이드는 얼굴 데이터셋에서 **홍조(Redness), 주름(Wrinkle), 모공(Pore)** 정보를 추출하여, 후속 작업인 **HighRes-Face-DeAging-SPADE** 모델 학습에 사용할 수 있는 **4채널 조건지도(Condition Maps)**를 생성하는 파이프라인 실행 매뉴얼

---

## 개발 환경 구축

```bash
conda create -n face-derma-condmaps python=3.10 -y
conda activate face-derma-condmaps

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
(설치 후 확인: python -c "import torch; print(torch.cuda.is_available())")

pip install -r requirements.txt
```

---

## 필수 모델 및 서브모듈 준비
- 이 파이프라인은 피부 영역 추출을 위해 **BiSeNet**을 사용하므로, 아래 두 가지 자산이 반드시 제 위치에 있어야 함
1. Face Parsing 소스 코드 (Submodule)
  - [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) 리포지토리의 코드가 필요
  - 프로젝트 클론 시 아래 명령어를 통해 서브모듈을 가져옴
  (또는 직접 다운로드하여 `third_party/face-parsing.PyTorch` 경로에 압축 해제)
  ```bash
  git submodule update --init --recursive
  ```

2. BiSeNet 사전학습 가중치
   - 파일명: `79999_iter.pth`
   - 다운로드 후 프로젝트 루트 하위에 `models/` 폴더를 생성하고 그 안에 넣음
   (`models/79999_iter.pth`)

---

### 디렉토리 구조

```plain text
face-derma-condmaps
├─ config
│  ├─ config.yaml
│  └─ config_highdetail.yaml
├─ detectors
│  └─ insightface_detector.py
├─ models                # (직접 생성) 79999_iter.pth 배치
├─ README.md
├─ requirements.txt
├─ scripts               # 보조 스크립트 (통계, 데이터 분할 등)
│  ├─ dryrun_dataloader.py
│  ├─ make_splits.py
│  ├─ mosaic_from_list.py
│  ├─ run_pipeline.bat
│  ├─ stats_report.py
│  └─ test_bisenet_skin.py
├─ src                   # 핵심 소스 코드
│  ├─ metrics
│  │  ├─ pore.py
│  │  ├─ redness.py
│  │  ├─ wrinkle.py
│  │  └─ __init__.py
│  ├─ parsing
│  │  ├─ bisenet_parser.py
│  │  └─ __init__.py
│  ├─ pipeline
│  │  ├─ run_pipeline.py
│  │  └─ __init__.py
│  ├─ preprocess_pipeline.py
│  ├─ qc
│  │  ├─ qc_filter.py
│  │  ├─ quality.py
│  │  └─ __init__.py
│  ├─ utils
│  │  ├─ io.py
│  │  ├─ viz.py
│  │  └─ __init__.py
│  └─ __init__.py
└─ third_party           # (서브모듈) face-parsing.PyTorch
```

---

## 파이프라인 환경 설정 (`config.yaml`)
- 실행 전 `config/config.yaml`(또는 `config_highdetail.yaml`) 파일을 열어 로컬 환경에 맞춰 입출력 경로 지정
```YAML
paths:
  image_root: "원본 이미지 폴더 절대/상대 경로"   # (예: D:/data/raw_faces)
  output_root: "결과물이 저장될 폴더 경로"         # (예: D:/data/processed)
  bisenet_ckpt: "models/79999_iter.pth"          # 다운로드한 BiSeNet 가중치 경로
```
*1024px의 고해상도 환경에서 미세한 잔주름과 모공 텍스처를 더 정밀하게 추출하고 싶다면 `config/config_highdetail.yaml`을 사용하시기 바랍니다.*

---

## 파이프라인 실행 가이드

### 1. SPADE용 조건지도 생성
- 원본 이미지에서 얼굴을 정렬 및 크롭하고, **High-Res-DeAging-SPADE** 모델의 입력 규격에 호환되는 **[홍조, 주름, 모공, 마스크] 4채널 데이터**를 생성
```bash
python src/preprocess_pipeline.py --config config/config.yaml
```

- CSV 기반 수치형 품질 분석 지표(`labels.csv`)나 채널별 세부 분리 파일 필요 시, 아래 스크립트 대신 실행
```bash
python -m src.pipeline.run_pipeline --config config/config.yaml
```

### 2. 데이터 품질 점검(QC) 및 통계 (선택)
*(위 참고 명령어인 run_pipeline.py를 실행하여 labels.csv가 생성된 경우에 수행 가능)*
추출된 데이터 중 심하게 흔들리거나 어두운 불량 이미지를 수치 기반으로 필터링

1. **통계 리포트 생성**: 점수 분포와 상관관계를 시각화하여 이상치 기준 선정
```bash
python scripts/stats_report.py --labels [출력경로]/labels.csv --out [출력경로]/reports
```

2. 데이터 필터링: 품질 기준(Blur, 노출, 마스크 면적 등)을 통과한 데이터 선정
```bash
python src/qc/qc_filter.py --config config/config.yaml --labels [출력경로]/labels.csv --out [출력경로]/qc --copy
```

### 3. 데이터셋 학습/검증 분할 (선택)
- 품질 검증을 통과한 데이터(pass.csv)를 기반으로 Train / Val 셋을 분할하고 메타데이터가 포함된 매니페스트(manifest.jsonl)를 생성
```bash
python scripts/make_splits.py --config config/config.yaml --pass_csv [출력경로]/qc/pass.csv --out [출력경로]/splits --val_ratio 0.1
```

## 라이선스/출처

- **레포지토리**: MIT License
- **사용 모델/코드**:
    - InsightFace (MIT License)
    - BiSeNet (CelebA-HQ 파생, 가중치 `79999_iter.pth`는 사용자 소지 파일)
- **데이터**: FFHQ(NVLabs) - 데이터 라이선스/정책 준수

---
