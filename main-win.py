import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os


os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['OMP_NUM_THREADS'] = '1' 
class WindowsObjectDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu" and torch.cuda.is_available()
        
        # Load optimized YOLO model
        self.model = YOLO("yolov8s.pt").to(self.device)
        if self.half:
            self.model.half()
            print("Using FP16 precision")
        else:
            print("Using FP32 precision")


        self.config = {
            "conf_thres": 0.4,
            "iou_thres": 0.3,
            "imgsz": 640,
            "tile_size": 320,
            "small_obj_size": 32,
            "frame_skip": 2,
            "backend": cv2.CAP_DSHOW,
            "gpu_threads": 4 if self.half else 0
        }


        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_num_threads(self.config["gpu_threads"])

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
                    verbose=False,
                    half=self.half
                )
                
                for box in results[0].boxes:
                    if self._is_small_object(box.xywh[0]):
                        adjusted = box.xyxy.clone()
                        adjusted[0][0::2] += x
                        adjusted[0][1::2] += y
                        detections.append({
                            'xyxy': adjusted[0].cpu().numpy(),
                            'conf': box.conf.item(),
                            'cls': int(box.cls.item())
                        })
        return detections

    def _is_small_object(self, box):
        _, _, width, height = box.cpu().numpy()
        return max(width, height) < self.config["small_obj_size"]

    def _windows_draw(self, frame, boxes):
        """Optimized drawing for Windows DirectShow"""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.model.names[box['cls']]} {box['conf']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def process_frame(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        full_dets = self.model.predict(
            rgb_frame,
            imgsz=self.config["imgsz"],
            conf=self.config["conf_thres"],
            iou=self.config["iou_thres"],
            device=self.device,
            verbose=False,
            half=self.half
        )
        

        combined = [{
            'xyxy': box.xyxy[0].cpu().numpy(),
            'conf': box.conf.item(),
            'cls': int(box.cls.item())
        } for box in full_dets[0].boxes] if full_dets[0].boxes else []
        
        combined += self._tiled_detection(rgb_frame)
        return self._windows_draw(frame.copy(), combined)

def main():
    detector = WindowsObjectDetector()
    cap = None
    
    # Windows camera initialization with fallback
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, detector.config["backend"])
        if cap.isOpened():
            print(f"Connected to camera index {idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            break
    else:
        print("Error: No cameras detected")
        return

    cv2.namedWindow("Object Detection by Goutham", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection by Goutham", 800, 600)

    try:
        frame_count = 0
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
            cv2.imshow("Object Detection by Goutham", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()