import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

class SmallObjectDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing detector on {self.device}")

        self.model = YOLO("yolov8s.pt").to(self.device)
        
        self.config = {
            "conf_thres": 0.4,
            "iou_thres": 0.3,
            "imgsz": 640,
            "tile_size": 320,
            "small_obj_size": 32,
            "frame_skip": 2
        }

    def _tiled_detection(self, frame):
        h, w = frame.shape[:2]
        detections = []
        
        for y in range(0, h, self.config["tile_size"]):
            for x in range(0, w, self.config["tile_size"]):
                tile = frame[y:y+self.config["tile_size"], x:x+self.config["tile_size"]]
                results = self.model.predict(
                    tile,
                    imgsz=self.config["imgsz"],
                    conf=self.config["conf_thres"],
                    iou=self.config["iou_thres"],
                    device=self.device,
                    verbose=False
                )
                
                for box in results[0].boxes:
                    if self._is_small_object(box.xywh[0]):
                        adjusted_box = box.xyxy.clone()
                        adjusted_box[0][0::2] += x
                        adjusted_box[0][1::2] += y
                        detections.append({
                            "xyxy": adjusted_box[0].cpu().numpy(),
                            "conf": box.conf.item(),
                            "cls": int(box.cls.item())
                        })
        return detections

    def _is_small_object(self, box):
        _, _, width, height = box.cpu().numpy()
        return max(width, height) < self.config["small_obj_size"]

    def _draw_boxes(self, frame, boxes):
        """Custom box drawing function to replace .plot() method"""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            confidence = box["conf"]
            class_id = box["cls"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{self.model.names[class_id]} {confidence:.2f}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return frame

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        full_dets = self.model.predict(
            rgb_frame,
            imgsz=self.config["imgsz"],
            conf=self.config["conf_thres"],
            iou=self.config["iou_thres"],
            device=self.device,
            verbose=False
        )
        
       
        full_boxes = [{
            "xyxy": box.xyxy[0].cpu().numpy(),
            "conf": box.conf.item(),
            "cls": int(box.cls.item())
        } for box in full_dets[0].boxes] if full_dets[0].boxes else []
        
        tile_boxes = self._tiled_detection(rgb_frame)
        

        combined_boxes = full_boxes + tile_boxes
        return self._draw_boxes(frame.copy(), combined_boxes)

def main():
    detector = SmallObjectDetector()
    frame_count = 0


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    
    try:
        while cap.isOpened():
            frame_count += 1
            if frame_count % detector.config["frame_skip"] != 0:
                cap.grab()
                continue

            ret, frame = cap.read()
            if not ret:
                print("Frame capture error")
                break

            processed = detector.process_frame(frame)
            cv2.imshow("Object Detection", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()