import cv2
import pandas as pd
import streamlit as st
from yt_dlp import YoutubeDL
from tracker import ObjectTracker
from ui import TrackerUI

# Streamlit 페이지 설정
st.set_page_config(page_title="지능형 객체 추적 시스템", layout="wide")

def is_youtube_url(url):
    """유튜브 영상 주소인지 간단히 확인"""
    url = url.strip().lower()
    return any(host in url for host in ("youtube.com/", "youtu.be/"))

def download_youtube_video(url, output_path="temp_youtube.mp4"):
    """유튜브 URL에서 영상을 안정적으로 다운로드하는 함수"""
    ydl_opts = {
        # 추적에는 소리가 필요 없으므로 영상 스트림만 받음 (최근 유튜브는 영상+소리 합본 포맷이 없는 경우가 많음)
        # OpenCV가 읽을 수 있는 H.264(avc1) mp4, 720p 이하 우선 → 없으면 차선책으로 대체
        'format': (
            'bestvideo[ext=mp4][vcodec^=avc1][height<=720]'
            '/bestvideo[ext=mp4][height<=720]'
            '/bestvideo[height<=720]'
            '/best'
        ),
        'outtmpl': output_path,
        'overwrites': True,
        'noplaylist': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def main():
    # UI 컴포넌트 로드
    model_type, tracker_type, conf, iou, input_source, uploaded_file, yt_url = TrackerUI.setup_sidebar()
    video_placeholder, metrics_placeholder = TrackerUI.setup_main_layout()

    target_video_path = None
    run_tracking = False

    # 1. 파일 업로드 모드인 경우
    if input_source == "파일 업로드":
        if uploaded_file is not None:
            target_video_path = "temp_input.mp4"
            with open(target_video_path, "wb") as f:
                f.write(uploaded_file.read())
            
            if st.sidebar.button("🚀 추적 시작", type="primary"):
                run_tracking = True

    # 2. 유튜브 URL 입력 모드인 경우 (다운로드 후 원클릭 실행)
    else:
        if yt_url:
            if st.sidebar.button("🚀 유튜브 추적 시작", type="primary"):
                with st.spinner("유튜브 영상 다운로드 및 분석 준비 중... 잠시만 기다려주세요!"):
                    try:
                        target_video_path = download_youtube_video(yt_url)
                        run_tracking = True
                        st.sidebar.success("다운로드 완료! 추적을 시작합니다.")
                    except Exception as e:
                        st.sidebar.error(f"다운로드 실패: {e}")

    # 추적 실행 플래그가 켜진 경우 MOT 구동
    if run_tracking and target_video_path:
        tracker = ObjectTracker(model_path=model_type, tracker_type=tracker_type)
        cap = cv2.VideoCapture(target_video_path)
        
        frame_count = 0
        tracking_logs = []
        
        st.sidebar.info("객체 추적이 진행 중입니다...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 프레임별 객체 검출 및 추적
            annotated_frame, current_fps, logs = tracker.process_frame(frame, conf, iou)
            
            # 로그 누적
            for log in logs:
                log["Frame"] = frame_count
                tracking_logs.append(log)
            
            # 화면 출력
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # 실시간 지표 업데이트
            with metrics_placeholder.container():
                st.metric(label="현재 처리 속도 (FPS)", value=f"{current_fps:.2f} FPS")
                st.write(f"진행된 프레임 번호: {frame_count}")
                st.write(f"현재 프레임 검출 객체 수: {len(logs)}개")

        cap.release()
        st.sidebar.success("추적이 완료되었습니다!")
        
        # CSV 로그 다운로드 버튼
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
        if input_source != "파일 업로드" and yt_url:
            # 유튜브 URL이 입력되면 추적 전에 원본 영상을 바로 재생할 수 있도록 미리보기 표시
            if is_youtube_url(yt_url):
                with video_placeholder.container():
                    st.video(yt_url)
                    st.caption("▶️ 원본 영상 미리보기 — 사이드바의 '유튜브 추적 시작'을 누르면 객체 추적을 시작합니다.")
            else:
                st.warning("올바른 유튜브 영상 주소를 입력해주세요. (예: https://www.youtube.com/watch?v=...)")
        elif not target_video_path:
            st.info("왼쪽 사이드바에서 동영상 파일을 업로드하거나 유튜브 URL을 입력해주세요.")

if __name__ == "__main__":
    main()