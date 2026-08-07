# 치안 바디캠 합성 데이터 생성 파이프라인

치안 비정형데이터 활용을 위한 암호화 AI 모델 검증용 합성 바디캠 이미지 데이터 생성 시스템

### 동형암호 얼굴 매칭 파이프라인

```
바디캠 영상 → 얼굴 검출(RetinaFace) → 512차원 임베딩(ArcFace) → HE 암호화 → 암호화 유사도 검색 → 경보
```

---

## 전체 파이프라인 구조

```
[Stage 1] 얼굴 ID 풀 생성
    DeceiveD (4_Cond_Exp, 무표정)
        ↓ 200개 고유 얼굴 생성 (1024×1024)
    InsightFace buffalo_l (ArcFace R100)
        ↓ 512차원 임베딩 추출 + 성별/나이 예측
    Farthest-Point Sampling
        ↓ 최다양 20개 ID 선별
    data/celeb-k/face_pool_controlled/selected/

[Stage 2] 바디캠 씬 생성
    IP-Adapter-FaceID + RealVisXL V5.0 (SDXL)
        ↓ 선별된 얼굴 임베딩 → 씬 내 얼굴 ID 고정
    Scene-Sequential Generation
        ↓ 씬당 10프레임 (프레임 0: text2img, 1-9: img2img 강도 0.35)
    8,400장 합성 바디캠 이미지 (20 ID × 420장)

[Stage 3] 자동 어노테이션
    YOLOv8x + Grounding-DINO → 객체 검출
    SAM2 → 세그멘테이션
    COCO 포맷 저장 (scene_id, frame_index 포함)

[Stage 4] 품질 검증
    FID/KID, CLIP Score, SSIM/LPIPS, 얼굴 검출률
```

---

## 대상 시나리오

### 시나리오 1: 관리 대상자/수배자 식별

경찰관이 바디캠을 착용하고 한국 도심 지역을 순찰하며 대상자를 식별합니다.

- 대상자 얼굴 검출 및 임베딩 추출
- 암호화된 수배자 DB와 유사도 검색
- 무기 소지 등 위협 상황 포함

### 시나리오 2: 실종자 탐지

아동, 고령자, 치매 환자 등 실종 대상자를 바디캠을 통해 탐지합니다.

- 실종 아동/고령자 얼굴 매칭
- 외모 기반 유사도 검색

---

## 얼굴 ID 풀 (Face Identity Pool)

### 모델: DeceiveD (StyleGAN2 + APA)

**Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data** (NeurIPS 2021)  
Celeb-K 한국인 얼굴 데이터셋으로 훈련된 1024×1024 얼굴 생성 모델

| 체크포인트 | FID | 조건 |
|-----------|-----|------|
| 1_Uncond | 14 | 무조건 생성 |
| 2_Cond_Gender | 18 | 성별 조건 |
| 3_Cond_Yaw | 18 | 얼굴 각도 조건 |
| 4_Cond_Exp | 18 | 표정 조건 (7종) |

### 표정 클래스 (4_Cond_Exp)

| class | 표정 |
|-------|------|
| 0 | happy (행복) |
| 1 | surprise (놀람) |
| 2 | neutral (무표정) ← 현재 사용 |
| 3 | disgust (혐오) |
| 4 | angry (분노) |
| 5 | fear (공포) |
| 6 | sad (슬픔) |

### 현재 생성 결과 (`data/celeb-k/face_pool_controlled/`)

| 항목 | 값 |
|------|-----|
| 생성 모델 | DeceiveD 4_Cond_Exp (무표정, class=2) |
| 생성 수 | 200개 고유 ID |
| 얼굴 검출률 | 100% |
| 성별 | Male 119 / Female 81 |
| 평균 나이 | 30.8세 (±8.7) |
| 코사인 유사도 평균 | 0.223 (낮을수록 다양) |
| 근접 중복 쌍 (>0.7) | 3 / 19,900 |
| 최종 선별 ID | 20개 (farthest-point sampling) |
| VRAM 사용량 | ~122 MB |
| 생성 속도 | ~64ms/장 (TITAN RTX) |

### EDA 노트북

`data/celeb-k/face_pool_controlled/` 및 `data/celeb-k/face_pool/face_pool_eda.ipynb`

분석 항목:
- 생성 얼굴 시각적 검사
- ArcFace 임베딩 품질 (L2 norm, 분포)
- 쌍별 코사인 유사도 분포 + CDF
- PCA / t-SNE 신원 공간 시각화
- 임베딩 매트릭스 히트맵
- 선별 20개 ID 비교
- 성별/나이 인구통계 분석

---

## 프로젝트 구조

```
├── src/
│   ├── config.py                  # 파이프라인 설정 (YAML → dataclass)
│   ├── generate/
│   │   ├── text2img.py            # RealVisXL 씬 생성 (text2img + img2img 체이닝)
│   │   ├── identity.py            # 얼굴 ID 풀 로드 및 씬 배정
│   │   └── vram.py                # VRAM 관리 (모델 스코프)
│   ├── prompts/
│   │   ├── templates.py           # 바디캠 프롬프트 템플릿 (1인칭 시점)
│   │   ├── scene_planner.py       # 씬 순차 프롬프트 생성
│   │   └── scenario_engine.py     # 시나리오별 조합 생성
│   ├── annotate/
│   │   ├── detector.py            # Grounding-DINO 검출
│   │   ├── yolo_detector.py       # YOLOv8 검출
│   │   ├── segmentor.py           # SAM2 세그멘테이션
│   │   └── coco_formatter.py      # COCO 포맷 변환 (scene_id, frame_index 포함)
│   └── validate/
│       ├── face_quality.py        # 얼굴 검출률 + 임베딩 품질
│       ├── fid_kid.py             # FID/KID
│       ├── clip_score.py          # CLIP Score
│       └── ssim_lpips.py          # 재식별 안전성
├── scripts/
│   ├── generate_face_pool.py          # DeceiveD 무조건 얼굴 풀 생성
│   ├── generate_face_pool_controlled.py  # DeceiveD 표정 조건 얼굴 풀 생성 ← 현재 사용
│   ├── eda_face_pool.py               # 얼굴 풀 EDA 스크립트
│   ├── run_generate.py                # 바디캠 씬 생성 실행
│   ├── run_annotate.py                # 어노테이션 실행
│   └── run_validate.py                # 검증 실행
├── config/
│   ├── default.yaml               # 기본 설정 (RealVisXL V5.0, guidance=7.5)
│   └── faceid_test.yaml           # FaceID 테스트 설정
├── data/
│   └── celeb-k/
│       ├── face_pool/             # 무조건 생성 얼굴 풀 (1_Uncond)
│       │   ├── generated/         # 200장 생성 이미지
│       │   ├── embeddings/        # ArcFace 임베딩 (.npy)
│       │   ├── selected/          # 선별 20개 ID
│       │   └── face_pool_eda.ipynb  # EDA 노트북
│       └── face_pool_controlled/  # 표정 조건 얼굴 풀 (4_Cond_Exp, 무표정)
│           ├── generated/         # 200장 무표정 이미지
│           ├── embeddings/        # ArcFace 임베딩 + 성별/나이 메타데이터
│           ├── selected/          # 선별 20개 ID
│           └── face_pool_report.json
└── notebooks/
```

---

## 씬 순차 생성 (Scene-Sequential Generation)

동일 씬 내 연속 프레임 간 일관성을 위해 img2img 체이닝 사용:

```
프레임 0: text2img (씬 기준 프레임, scene_seed 고정)
프레임 1-9: img2img from 프레임 0 (강도=0.35)
    → 동일 장소/조명 유지, 피사체 행동 변화
```

- 누적 열화 방지: 프레임 N-1이 아닌 프레임 0에서 항상 디노이징
- 씬당 10프레임, 총 840씬 → 8,400장

---

## 사용 모델

| 용도 | 모델 |
|------|------|
| 얼굴 ID 생성 | DeceiveD (StyleGAN2+APA, Celeb-K 훈련) |
| 바디캠 씬 생성 | RealVisXL V5.0 (SDXL 파인튜닝) |
| 얼굴 ID 조건 | IP-Adapter-FaceID (SDXL) |
| 얼굴 검출 | InsightFace buffalo_l / RetinaFace |
| 얼굴 임베딩 | ArcFace R100 (512차원) |
| 객체 검출 | YOLOv8x, Grounding-DINO |
| 세그멘테이션 | SAM2 |
| 품질 평가 | CLIP ViT-H-14, LPIPS (AlexNet) |

---

## 평가 프레임워크

### 유용성 (Utility)
| 지표 | 설명 | 기준 |
|------|------|------|
| FID | 실제-합성 분포 유사도 | 낮을수록 좋음 |
| KID | 커널 기반 분포 유사도 | 낮을수록 좋음 |
| CLIP Score | 텍스트-이미지 정합도 | ≥ 0.2 |

### 안전성 (Safety)
| 지표 | 설명 | 기준 |
|------|------|------|
| SSIM | 구조적 유사도 (재식별 위험) | < 0.6 |
| LPIPS | 지각적 유사도 (재식별 위험) | > 0.3 |

### 얼굴 품질
| 지표 | 설명 | 기준 |
|------|------|------|
| 얼굴 검출률 | InsightFace 기반 검출 | ≥ 85% |
| 코사인 유사도 평균 | 임베딩 쌍별 분포 | 낮을수록 다양 |
| 근접 중복 쌍 비율 | cosine > 0.7 비율 | 0에 가까울수록 좋음 |

---

## 환경 설정

### 요구 사항
- Python 3.11+
- CUDA 12.1+ 호환 GPU (VRAM 24GB 권장)
- conda

### 설치

```bash
conda create -n police python=3.11 -y
conda activate police
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate transformers safetensors
pip install insightface onnxruntime-gpu==1.20.1
pip install scikit-learn scikit-image lpips open-clip-torch
pip install pillow numpy opencv-python pyyaml tqdm matplotlib
```

### 실행

```bash
# GPU 1 사용 (GPU 0은 디스플레이 전용)
export CUDA_VISIBLE_DEVICES=1

# 1. 얼굴 ID 풀 생성 (무표정, 200개)
python scripts/generate_face_pool_controlled.py --num-faces 200 --num-select 20

# 2. EDA 노트북 실행
jupyter notebook data/celeb-k/face_pool/face_pool_eda.ipynb

# 3. 바디캠 씬 생성
python scripts/run_generate.py --identity-ref-dir data/celeb-k/face_pool_controlled/selected
```
