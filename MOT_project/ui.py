import streamlit as st

class TrackerUI:
    @staticmethod
    def setup_sidebar():
        """사이드바 설정 및 사용자 입력 값 반환 (파일 및 유튜브 지원)"""
        st.sidebar.header("⚙️ 시스템 설정 및 제어")
        model_type = st.sidebar.selectbox("모델 선택", ["yolov8n.pt", "yolov8s.pt"])
        tracker_type = st.sidebar.selectbox("추적 알고리즘", ["bytetrack.yaml", "botsort.yaml"])

        conf_threshold = st.sidebar.slider("객체 검출 임계값 (Confidence)", 0.1, 0.9, 0.25, 0.05)
        iou_threshold = st.sidebar.slider("IoU 임계값 (Matching Threshold)", 0.1, 0.9, 0.7, 0.05)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎥 영상 소스 선택")
        input_source = st.sidebar.radio("입력 방식을 선택하세요:", ["파일 업로드", "유튜브 URL 입력"])
        
        uploaded_file = None
        yt_url = None

        if input_source == "파일 업로드":
            uploaded_file = st.sidebar.file_uploader("동영상 파일 업로드 (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        else:
            yt_url = st.sidebar.text_input("유튜브 영상 링크를 입력하세요:")
        
        return model_type, tracker_type, conf_threshold, iou_threshold, input_source, uploaded_file, yt_url

    @staticmethod
    def setup_main_layout():
        """메인 화면 레이아웃 구성"""
        st.title("🤖 지능형 객체 추적 시스템 (MOT)")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📹 영상 출력 및 궤적 시각화")
            video_placeholder = st.empty()
        with col2:
            st.subheader("📊 실시간 분석 및 지표")
            metrics_placeholder = st.empty()
            
        return video_placeholder, metrics_placeholder