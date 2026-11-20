# tracking_alert

사람·차량을 추적하고, **급감속 후 저속 유지**가 확인되면 사고 의심으로 알리는 프로그램입니다.

메인 화면에서 아래 모드를 고릅니다.

| 모드 | 설명 |
| --- | --- |
| 화면 캡처 | 모니터/영역을 잡아 실시간 추적 |
| 영상 파일 | 로컬·저장소 영상 분석 후 녹화본 검토 |
| 성능평가 | `eval_data` 사고/정상 영상으로 정확도 측정 |

## 실행

```powershell
cd tracking_alert
pip install -r requirements.txt
python ui_app.py
```

처음 실행 시 `yolov8n.pt`를 자동으로 받습니다.

## 화면 표시

| 표시 | 의미 |
| --- | --- |
| 초록 실선 | 일반 추적 |
| 빨간 실선 | 사고 의심 알림 |
| 주황 점선 | 가림(가려짐) 예측 |
| 화살표 | 속도 방향 |

설정은 오른쪽 패널에서 바꾸고, **현재 설정 저장**으로 `user_settings.json`에 남길 수 있습니다.  
**사고 알림음** 체크를 끄면 사고 확정 시 Windows 알림음만 음소거됩니다. (로그·박스 표시는 유지)

## 사고 판정 (요약)

속도는 박스 **중심 이동**에 더해, 박스가 작아지면 **앞으로(멀어짐)** 가는 것으로 보정합니다.  
중심 픽셀 속도가 거의 0이어도 크기가 줄어들면 계속 움직이는 것으로 봅니다.

1. 움직이던 객체의 속도가 크게 떨어짐  
2. 그 저속 상태가 잠시 유지됨 → `accident_suspect` (낙하 패턴이면 `fall_suspect`)

박스 겹침 기반 근접 충돌 태그는 쓰지 않습니다. (원근·세로 영상 오탐)

검출이 부족하면 **검출 신뢰도**를 낮추면 됩니다. (추적기 임계값도 같이 맞춰집니다.)  
코너 선회 제외 필터는 끄고, 박스 크기 원근 보정 속도 기준으로만 급감속을 봅니다.

## 성능평가

```text
eval_data/
  accident/   # 사고 영상
  normal/     # 정상 영상
```

UI의 **성능평가** 또는:

```powershell
python eval_accuracy.py --data eval_data
```

결과는 `outputs/eval_accuracy/`에 저장됩니다. (영상 파일은 git에 올리지 않습니다.)

## 다중 객체 추적(MOT) 성능 평가 · 개선

과제 요구사항(데이터 분할, 추적기 구현·개선, HOTA/MOTA/IDF1, FPS, Fine-tuning)을 위한 기능입니다.

### 추적기 (`src/bytetrack.py`, 직접 구현)

ByteTrack 처리 과정(칼만 예측 → high 검출 1차 연관 → low 검출 2차 연관 → 미확정 트랙 처리 → 유실/제거 → 신규 생성)을 직접 구현했고, 두 가지를 개선했습니다.

| 개선 | 내용 | 관련 요구사항 |
| --- | --- | --- |
| A. 외형 특징 융합 | 박스 안 HSV 색 히스토그램(Bhattacharyya 거리)을 IoU 비용과 가중합 | 특징 추출 · 연관성 판단 |
| B. 유실 트랙 재연결 | IoU로 못 붙은 새 검출을 Lost/최근 제거 트랙과 예측 중심 거리·크기·외형으로 재비교해 예전 ID 복원 | 이동 예측 · 추적 상태 관리 · 재등장 |

GUI의 **추적 알고리즘**에서 직접 구현(사고 감지용/최종 개선/개선/기본값)과 Ultralytics 내장(비교 기준)을 고를 수 있습니다.

### 데이터셋 · 평가

MOT16 공개 데이터셋(정답 박스·ID 포함)을 시퀀스 단위로 나눕니다.

| 분할 | 시퀀스 | 용도 |
| --- | --- | --- |
| train | MOT16-02, 04, 13 | YOLO Fine-tuning |
| val | MOT16-10, 11 | Fine-tuning 검증, 추적 파라미터 튜닝 |
| test | MOT16-05, 09 | 최종 비교 (튜닝·학습에 미사용) |

```powershell
# MOT16.zip(1.9GB)을 datasets/ 에 받은 뒤
python mot_eval.py prepare   # 압축 해제, 분할, YOLO 형식 변환(datasets/mot16_yolo)
python finetune.py --epochs 10   # → weights/yolov8n_mot16.pt (+ .json 학습 기록)
python mot_eval.py tune      # val 격자 탐색 → configs/tuned_tracker.json
python mot_eval.py eval      # test 비교 → outputs/mot_eval/summary.json / summary.csv
```

지표: HOTA(DetA/AssA, 직접 구현), MOTA, IDF1, ID Switch, FP/FN, MT/ML, Frag(motmetrics), 프레임당 처리 시간·FPS.
메인 화면 **MOT 성능 비교**에서 같은 작업을 버튼으로 실행하고 결과 표·그래프를 볼 수 있습니다.

### 궤적 · 결과 저장/불러오기

- 추적 중 객체별 이동 경로(ID 색 선), 이동 방향(8방위), 누적 이동 거리, 평균 속도를 계산·표시
- 영상 추적이 끝나면 `outputs/review/last_tracks.csv`(프레임·ID·좌표), `last_tracks.mot.txt`(MOT 형식), `last_meta.json`(모델·추적기 설정·성능·궤적 요약) 저장
- **추적 결과 불러오기**로 저장된 CSV를 원본 영상 위에 다시 그려 재생 (재추적 없음)

## 기타 실행

```powershell
python main.py --source synthetic   # 합성 영상 테스트
python main.py --source screen      # 화면 전체 추적
```

## AI 사용

이 프로젝트는 Cursor(AI)로 코드 작성·디버깅·UI/평가 기능 보강을 했으며, 판정 로직과 실험 설정은 직접 확인하고 조정했습니다.
