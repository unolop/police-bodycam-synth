# 치안 바디캠 합성 데이터 생성 파이프라인

치안 비정형데이터 활용을 위한 암호화 AI 모델 검증용 합성 바디캠 이미지 데이터 생성 시스템

## 프로젝트 개요

본 프로젝트는 동형암호(HE) 기반 치안 AI 알고리즘의 개발 및 검증을 지원하기 위한 **합성 바디캠 이미지 데이터셋**을 생성합니다.

실제 바디캠 영상 데이터의 수집이 어렵고 개인정보 이슈가 있기 때문에, 텍스트-이미지 생성 모델(SDXL)을 활용하여 사실적인 합성 데이터를 생성하고, 이를 동형암호 기반 얼굴 매칭 파이프라인의 검증에 활용합니다.

### 동형암호 얼굴 매칭 파이프라인

```
바디캠 영상 → 얼굴 검출 → 512차원 임베딩 추출 → HE 암호화 → 암호화 유사도 검색 → 경보
```

## 대상 시나리오

### 시나리오 1: 관리 대상자/수배자 식별

경찰관이 바디캠을 착용하고 유흥가, 공공시설, 범죄 다발 지역 등을 순찰하며 대상자를 식별합니다.

- 대상자 얼굴 검출 및 임베딩 추출
- 암호화된 수배자 DB와 유사도 검색
- 무기 소지 등 위협 상황 포함

### 시나리오 2: 실종자 탐지

아동, 고령자, 치매 환자 등 실종 대상자를 바디캠을 통해 탐지합니다.

- 실종 아동/고령자 얼굴 매칭
- 외모 기반 유사도 검색

## 프로젝트 구조

```
├── src/
│   ├── config.py                  # 파이프라인 설정 (YAML → dataclass)
│   ├── generate/                  # 이미지 생성 모듈
│   │   ├── text2img.py            # SDXL 텍스트-이미지 생성
│   │   ├── conditioned.py         # ControlNet 조건부 생성
│   │   ├── faceid.py              # IP-Adapter FaceID 기반 생성
│   │   ├── identity.py            # 얼굴 ID 관리
│   │   ├── preprocessing.py       # 전처리
│   │   └── vram.py                # VRAM 관리
│   ├── prompts/                   # 프롬프트 생성 엔진
│   │   ├── templates.py           # 바디캠 프롬프트 템플릿
│   │   ├── scenario_engine.py     # 시나리오별 조합 생성
│   │   ├── scene_planner.py       # 씬 시퀀스 생성
│   │   └── prompt_store.py        # 프롬프트 저장/로드
│   ├── annotate/                  # 자동 어노테이션
│   │   ├── detector.py            # Grounding-DINO 검출
│   │   ├── yolo_detector.py       # YOLO 검출
│   │   ├── segmentor.py           # SAM2 세그멘테이션
│   │   ├── face_embedder.py       # 얼굴 임베딩 추출
│   │   ├── merger.py              # 어노테이션 병합
│   │   └── coco_formatter.py      # COCO 포맷 변환
│   ├── validate/                  # 품질 검증
│   │   ├── face_quality.py        # 얼굴 검출률/임베딩 품질
│   │   ├── ssim_lpips.py          # SSIM/LPIPS 안전성 평가
│   │   ├── fid_kid.py             # FID/KID 분포 유사도
│   │   ├── clip_score.py          # CLIP 텍스트-이미지 정합도
│   │   └── report.py              # 평가 리포트 생성
│   ├── extract/                   # 영상 프레임 추출
│   └── dataset/                   # 데이터셋 패키징
├── scripts/                       # 실행 스크립트
│   ├── generate_sequential.py     # 순차적 얼굴→시나리오 생성 (메인)
│   ├── run_evaluation.py          # 평가 실행
│   ├── run_overnight_test.py      # 야간 대량 생성 테스트
│   ├── run_generate.py            # 단일 생성 실행
│   ├── run_pipeline.py            # 전체 파이프라인 실행
│   ├── run_annotate.py            # 어노테이션 실행
│   ├── run_validate.py            # 검증 실행
│   ├── run_extract.py             # 프레임 추출 실행
│   └── compare_models.py          # 모델 비교
├── config/                        # YAML 설정 파일
│   ├── default.yaml               # 기본 설정
│   ├── scenario_poi.yaml          # 시나리오 1 설정
│   ├── scenario_missing.yaml      # 시나리오 2 설정
│   └── faceid_test.yaml           # FaceID 테스트 설정
├── notebooks/                     # EDA 노트북
└── docs/                          # 프로젝트 문서
```

## 생성 파이프라인

### 2단계 순차 생성 (`scripts/generate_sequential.py`)

**Phase 1 — 얼굴 초상화 생성**
- 얼굴 ID 데이터셋(62명)의 각 신원별 바디캠 스타일 초상화 생성
- 신원당 3장, 다양한 장소/시간대/조명 조건

**Phase 2 — 시나리오 액션 생성**
- 각 신원별 시나리오 액션 이미지 생성
- 10개 액션 세트:
  - `s1_poi_walking` — 대상자 접근
  - `s1_poi_standing` — 대상자 서성거림
  - `s1_poi_confrontation` — 대치 상황
  - `s1_poi_weapon_knife` — 칼 소지
  - `s1_poi_weapon_bat` — 야구배트 소지
  - `s1_poi_weapon_bottle` — 깨진 병 소지
  - `s1_poi_weapon_pipe` — 파이프 소지
  - `s1_poi_abandoned_bag` — 의심 가방
  - `s2_missing_child` — 실종 아동
  - `s2_missing_elderly` — 실종 고령자

### 실행 방법

```bash
# 환경 활성화
conda activate police

# 드라이 런 (프롬프트만 생성, 이미지 생성 안 함)
python scripts/generate_sequential.py --dry-run

# Phase 1만 실행 (얼굴 초상화)
python scripts/generate_sequential.py --phase 1

# Phase 2만 실행 (시나리오 액션)
python scripts/generate_sequential.py --phase 2

# 전체 실행 (야간 실행 권장)
python scripts/generate_sequential.py --phase both
```

## 평가 프레임워크

PPT 기반 합성 데이터 평가 체계를 따릅니다:

### 유용성 (Utility)
| 지표 | 설명 | 기준 |
|------|------|------|
| FID | 실제-합성 분포 유사도 | 낮을수록 좋음 |
| KID | 커널 기반 분포 유사도 | 낮을수록 좋음 |
| CLIP Score | 텍스트-이미지 정합도 | ≥ 0.2 |

### 안전성 (Safety)
| 지표 | 설명 | 기준 |
|------|------|------|
| SSIM | 구조적 유사도 (재식별 위험) | < 0.6 (안전) |
| LPIPS | 지각적 유사도 (재식별 위험) | > 0.3 (안전) |

### 얼굴 품질 (Face Quality)
| 지표 | 설명 | 기준 |
|------|------|------|
| 얼굴 검출률 | InsightFace 기반 검출 | ≥ 85% |
| 코사인 유사도 | 임베딩 쌍별 분포 | 낮을수록 다양 |
| 고유사도 쌍 비율 | cosine > 0.7 비율 | 0에 가까울수록 좋음 |

### 평가 실행

```bash
# 단일 디렉토리 평가
python scripts/run_evaluation.py output/sequential_gen/phase1_faces

# 전체 시나리오 평가
python scripts/run_evaluation.py output/sequential_gen --all-scenarios

# 참조 데이터셋 대비 평가 (FID 계산)
python scripts/run_evaluation.py output/sequential_gen/phase1_faces --reference-dir data/face_id_dataset
```

## 최근 생성 결과

**1,220장 생성 (Phase 1: 183장 + Phase 2: 1,037장)**

| 액션 세트 | 얼굴 검출률 | 상태 |
|-----------|------------|------|
| 얼굴 초상화 | 100.0% | PASS |
| 대상자 접근 | 98.4% | PASS |
| 대상자 서성 | 98.4% | PASS |
| 대치 상황 | 96.7% | PASS |
| 칼 소지 | 98.4% | PASS |
| 야구배트 소지 | 97.5% | PASS |
| 깨진 병 소지 | 100.0% | PASS |
| 파이프 소지 | 100.0% | PASS |
| 의심 가방 | 70.5% | FAIL |
| 실종 아동 | 100.0% | PASS |
| 실종 고령자 | 100.0% | PASS |
| **전체** | **97.5%** | |

## 사용 모델

| 용도 | 모델 |
|------|------|
| 이미지 생성 | RealVisXL V5.0 (SDXL 기반) |
| 얼굴 검출/임베딩 | InsightFace buffalo_l |
| 객체 검출 | YOLOv8x, Grounding-DINO |
| 세그멘테이션 | SAM2 |
| 품질 평가 | CLIP ViT-H-14, LPIPS (AlexNet) |

## 환경 설정

### 요구 사항
- Python 3.11+
- CUDA 12.1+ 호환 GPU (VRAM 24GB 권장)
- conda 환경

### 설치

```bash
conda create -n police python=3.11 -y
conda activate police
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate transformers compel safetensors
pip install insightface onnxruntime-gpu
pip install scikit-image lpips open-clip-torch
pip install pillow numpy opencv-python pyyaml tqdm
```

## 참여 기관

본 프로젝트는 **"치안 비정형데이터 이용 활성화를 위한 암호화 AI모델 개발"** 과제의 일환입니다.

- **서울여자대학교**: [B2] 치안활용 이미지 공개데이터셋 분석 + 합성데이터 생성
- **CryptoLab**: 동형암호 기반 얼굴 매칭 파이프라인 개발

### KPI 목표
- Phase 1: 합성 이미지 50,000장 이상 (카테고리당 1,000장)
- Phase 2: 합성 이미지 100,000장 이상 (카테고리당 2,000장)
- 유사도 검색 정확도: 80% (Phase 1), 85% (Phase 2)
- 임베딩 정확도: 평문 대비 ≥ 90% (Phase 1), ≥ 95% (Phase 2)
