# 📱 비지도 이상탐지 기반 중고 스마트폰 자동 등급·가격 산정 시스템

> **"결함을 외우지 말고, 정상을 이해하라"** — 정상만 학습해서 처음 보는 결함까지 잡습니다.

PatchCore 기반 **비지도 이상탐지(Unsupervised Anomaly Detection)** 로 중고폰 사진 1장에서
결함 유무·위치·심각도를 판정하고, 이를 등급(S~C)과 예상 거래가로 변환하는 캡스톤 프로젝트입니다.

캡스톤 프로젝트 · 팀 양념치킨(8조)

---

## 🎯 핵심 아이디어

기존 결함 탐지는 대부분 **객체탐지(YOLO)** 로 접근합니다. 하지만 결함을 일일이 라벨링해야 하고,
학습 때 본 결함만 잡으며, 클래스 불균형에 취약합니다. 우리 데이터가 정확히 이 한계에 부딪혔습니다.

그래서 문제를 **이상탐지**로 재정의했습니다. "정상 폰이 어떻게 생겼는가"만 학습하고,
거기서 벗어나는 모든 것을 결함으로 봅니다.

| 문제 | YOLO 방식 | 우리 방식 (PatchCore) |
|---|---|---|
| 결함 데이터 부족 | 라벨된 결함 다량 필요 → 치명적 | **결함 라벨 0장** → 문제 소멸 |
| 클래스 불균형 | 소수 클래스 학습 실패 | **클래스 개념 자체가 없음** |
| 처음 보는 결함 | 미학습 = 미탐지 | **본 적 없어도 탐지** (open-set) |
| 결함 위치 마스크 | 픽셀 마스크 수작업 | **마스크 없이 히트맵 자동** |

---

## 🧩 시스템 구조

```
정상 사진들 (결함 라벨 0장) ──▶ 메모리뱅크 (정상 패치의 기억)
                                      │
폰 사진 1장 ──▶ 이상 점수 계산 ──▶ 결함 히트맵 ──▶ 등급·예상가 (S A B C)
                (정상과의 거리)      (위치·심각도)
```

- **점선 = 학습 경로**: 정상 사진으로 1회 메모리뱅크 구축 (백본 동결, 역전파 없음)
- **실선 = 추론 경로**: 거래마다 NN 거리 계산만 수행

이상 점수: `s(x) = max_p min_(m∈M) ‖ φ_p(x) − m ‖₂`
(M: coreset 메모리뱅크, φ_p: 위치 p의 패치 특징)

---

## 📊 주요 결과 (예비 검증: 정상 21 + 결함 79)

| 지표 | 값 |
|---|---|
| Image-level AUROC (PatchCore) | **0.901** |
| PaDiM (비교군) | 0.963 |
| 앙상블 z-융합 (w*=0.4) | **0.977** |
| 결함 탐지율 (임계값 2.80) | 60 / 79 |
| 정상 정확도 · 정밀도 | 18/21 · 95.2% |
| 추론 시간 / 처리량 (Colab T4) | 102 ms/장 · 9.8 FPS |
| 메모리뱅크 크기 (coreset 10%) | 8,294 벡터 |

- DeLong test: 앙상블 vs PatchCore Δ+0.070, p=0.0023 (유의)
- **Open-set**: 학습에 없던 디스플레이 줄·번짐 결함도 점수 4.1~4.3으로 임계값을 크게 초과

> 한계도 투명하게 공개합니다: 반사·그림자로 인한 오탐(FP), 작은 결함의 미탐(FN),
> 스튜디오↔실매물 도메인 갭. 자세한 에러 분석은 노트북 §6-D 참고.

---

## ⚙️ 설치 및 실행

### Google Colab (권장)
이 프로젝트는 Colab T4 환경에서 개발되었습니다. 노트북을 Colab에서 열고 위에서부터 순서대로 실행하면 됩니다.

```bash
# 노트북 첫 셀에서 자동 설치됨
pip install torch torchvision scikit-learn scipy matplotlib opencv-python-headless gradio
```

### 로컬 실행
```bash
git clone https://github.com/<your-id>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook phone_anomaly_detection.ipynb
```

> ⚠️ 노트북 §1·§2는 `from google.colab import drive` 등 Colab 전용 코드를 포함합니다.
> 로컬 실행 시 해당 셀을 건너뛰고, 아래 데이터 경로만 본인 환경에 맞게 수정하세요.

### 데이터 경로 설정 (노트북 §2)
```python
GOOD_DIR   = "/content/drive/MyDrive/정상"   # 정상 폰 사진 폴더
DEFECT_DIR = "/content/drive/MyDrive/결함"   # 결함 폰 사진 폴더
```
폴더가 없으면 자동으로 합성 데모 데이터로 파이프라인이 동작합니다(`USE_DEMO`).

---

## 📁 노트북 구성

| 섹션 | 내용 |
|---|---|
| §1 | 환경 셋업 · 재현성 (seed 42 고정) |
| §2 | 데이터 재구성 (정상 / 결함 분리) |
| §3 | **PatchCore 밑바닥 구현** (라이브러리 없이 직접) |
| §4 | 메모리뱅크 학습 (정상 이미지만, coreset 10%) |
| §5 | 이상 탐지 + 위치 히트맵 |
| §6 | 정량 평가 — AUROC, 임계값 자동 탐색 |
| §6-A~K | Bootstrap CI · Ablation · PaDiM 비교 · 에러분석 · 효율 · Pixel-AUROC · 백본 비교 · t-SNE/PCA · 앙상블 · DeLong test · 민감도 분석 |
| §7 | YOLO vs Anomaly Detection 정량 비교 |
| §8 | Open-set 데모 (학습 때 안 본 결함) |
| §9 | 이상점수 → 등급/가격 산정 |
| §10 | Gradio 웹 데모 |

---

## 🛠 기술 스택

- **모델**: PatchCore (밑바닥 구현), PaDiM (비교군), 앙상블 z-융합
- **백본**: WideResNet50 (ImageNet 사전학습, layer2·3, 동결)
- **프레임워크**: PyTorch · scikit-learn · SciPy
- **데모**: Gradio
- **환경**: Google Colab T4, 입력 해상도 256×256

---

## 👥 팀 양념치킨 (8조)

- **순선기** — 모델 구현 · 실험 설계 · 코드 작성
- **양원근** — 데이터 구축 · 데이터 수집 · 조장

---

## 📚 References

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*, CVPR 2022
- Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection*, ICPR 2021
- Jocher et al., *Ultralytics YOLOv8*, 2023

라이선스: 공개 데이터 CC BY 4.0 · 생성 이미지는 사용 도구 이용약관 확인
