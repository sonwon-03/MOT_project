# 평가 데이터

사고 영상과 정상 영상을 나눠 넣고 `eval_accuracy.py`로 정확도를 측정합니다.

```text
eval_data/
  accident/   # 사고 있는 영상 (현재 23개)
  normal/     # 사고 없는 영상 (현재 14개)
```

## 출처

- accident: [ACCIDENT](https://accidentbench.github.io/) 공개 CCTV 미리보기  
  (정면충돌, 추돌, 측면충돌, T-bone, 단독사고)
- normal: [kerberos-io/virtual-rtsp](https://github.com/kerberos-io/virtual-rtsp) 고속도로 CCTV 샘플

영상 파일은 git에 올리지 않습니다.

## 판정 기준 (영상 단위)

| 실제 \ 예측 | 알림 발생 | 알림 없음 |
|-------------|-----------|-----------|
| accident    | TP        | FN        |
| normal      | FP        | TN        |

지표: Precision, Recall, F1, Accuracy, False Alarm Rate

## 실행

```powershell
cd tracking_alert
python eval_accuracy.py --data eval_data --sensitivity 3
```

결과는 `outputs/eval_accuracy/`에 저장됩니다.
