import cv2
import pandas as pd
import streamlit as st
from tracker import ObjectTracker
from ui import TrackerUI

# Streamlit 페이지 설정
st.set_page_config(page_title="지능형 객체 추적 시스템", layout="wide")

def main():
    # UI 컴포넌트 로드
    model_type, tracker_type, conf, iou, uploaded_file = TrackerUI.setup_sidebar()
    video_placeholder, metrics_placeholder = TrackerUI.setup_main_layout()

    if uploaded_file is not None:
        temp_video_path = "temp_input.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        if st.sidebar.button("🚀 추적 시작 / 재시작"):
            # 객체 생성 (클래스 활용)
            tracker = ObjectTracker(model_path=model_type, tracker_type=tracker_type)
            cap = cv2.VideoCapture(temp_video_path)
            
            frame_count = 0
            tracking_logs = []
            
            st.sidebar.info("객체 추적이 진행 중입니다...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 트래커 클래스를 통한 프레임 처리
                annotated_frame, current_fps, logs = tracker.process_frame(frame, conf, iou)
                
                # 로그 데이터 누적
                for log in logs:
                    log["Frame"] = frame_count
                    tracking_logs.append(log)
                
                # 화면 출력 변환
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
                # 실시간 메트릭 업데이트
                with metrics_placeholder.container():
                    st.metric(label="현재 처리 속도 (FPS)", value=f"{current_fps:.2f} FPS")
                    st.write(f"진행된 프레임 번호: {frame_count}")
                    st.write(f"현재 프레임 검출 객체 수: {len(logs)}개")

            cap.release()
            st.sidebar.success("추적이 완료되었습니다!")
            
            # CSV 로그 다운로드 버튼 생성
            if tracking_logs:
                df_logs = pd.DataFrame(tracking_logs)
                csv_data = df_logs.to_csv(index=False).encode('utf-8-sig')
                st.sidebar.download_button(
                    label="📁 추적 결과 로그 다운로드 (CSV)",
                    data=csv_data,
                    file_name="tracking_results_log.csv",
                    mime="text/csv"
                )
    else:
        st.info("왼쪽 사이드바에서 분석할 동영상 파일을 업로드해주세요.")

if __name__ == "__main__":
    main()