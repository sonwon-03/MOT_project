# tracking_alert

화면에서 추적할 영역을 고른 뒤, 사람·차량을 추적하고 사고 의심 움직임을 알리는 프로그램입니다.

`MOT_project` 폴더의 Streamlit 앱과는 별도로 동작합니다.

## 실행

```powershell
cd tracking_alert
pip install -r requirements.txt
python ui_app.py
```

1. 왼쪽 미리보기로 화면을 확인합니다.
2. **영역 선택**으로 영상 칸만 드래그합니다. `Esc`는 선택 취소입니다.
3. **추적 시작**을 누르면 검출 박스와 영역 테두리가 표시됩니다.
4. **중지**를 누르면 추적이 끝납니다.

처음 실행하면 `yolov8n.pt` 모델을 자동으로 받습니다.

## 조작

| 키 | 동작 |
| --- | --- |
| F1 | 가속도 급변 조건 |
| F2 | 급변 후 상태 유지 조건 |
| F3 | 근접 시 충돌 태그 |
| F12 | 현재 분석 화면 저장 |
| F9 | 추적 종료 |

사고 의심이 확정되면 알림음이 나고, 로그는 `outputs/suspect_events.csv`에 저장됩니다.

## 합성 영상으로 로직만 확인

```powershell
python main.py --source synthetic
```

화면 전체를 제어 창 없이 추적하려면 `python main.py --source screen`을 사용합니다.
