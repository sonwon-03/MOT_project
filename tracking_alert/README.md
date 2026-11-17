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

1. 움직이던 객체의 속도가 크게 떨어짐  
2. 그 저속 상태가 잠시 유지됨  
3. (옵션) 박스가 겹치면 `collision_suspect` 태그

코너 선회처럼 **방향만 바뀌고 속도가 유지**되는 경우는 사고로 보지 않도록 필터가 있습니다.  
검출이 부족하면 **검출 신뢰도**를 낮추면 됩니다. (추적기 임계값도 같이 맞춰집니다.)

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

## 기타 실행

```powershell
python main.py --source synthetic   # 합성 영상 테스트
python main.py --source screen      # 화면 전체 추적
```

## AI 사용

이 프로젝트는 Cursor(AI)로 코드 작성·디버깅·UI/평가 기능 보강을 했으며, 판정 로직과 실험 설정은 직접 확인하고 조정했습니다.
