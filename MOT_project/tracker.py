import time
import cv2
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, model_path="yolov8n.pt", tracker_type="bytetrack.yaml"):
        """YOLO 모델 및 트래커 초기화"""
        self.model = YOLO(model_path)
        self.tracker_type = tracker_type

    def process_frame(self, frame, conf=0.25, iou=0.7):
        """프레임별 객체 검출 및 추적 수행"""
        start_time = time.time()
        
        # YOLO track 수행
        results = self.model.track(
            frame, 
            persist=True, 
            tracker=self.tracker_type, 
            conf=conf, 
            iou=iou,
            verbose=False
        )
        
        end_time = time.time()
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0.0
        
        # 시각화된 프레임 반환
        annotated_frame = results[0].plot()
        
        # 로그 수집용 데이터 추출
        logs = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            for box, track_id in zip(boxes, track_ids):
                logs.append({
                    "Track_ID": int(track_id),
                    "X1": box[0], "Y1": box[1], "X2": box[2], "Y2": box[3]
                })
                
        return annotated_frame, fps, logs