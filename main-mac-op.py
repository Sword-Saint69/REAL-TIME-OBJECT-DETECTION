import cv2
import torch
import numpy as np
import os
import time
from ultralytics import YOLO
import threading

os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

class SmallObjectDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8s.pt").to(self.device)
        
        if self.device == "cpu":
            torch.set_num_threads(torch.get_num_threads())
            self.config = {
                "conf_thres": 0.4,
                "iou_thres": 0.3,
                "imgsz": 320,
                "tile_size": 160,
                "small_obj_size": 32,
                "alert_classes": [0, 2, 3, 5, 7],
                "alert_threshold": 3,
                "sharpening_kernel": np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]),
                "hist_bins": 64,
                "target_brightness": 0.4
            }
        else:
            self.model.fuse()
            self.model.half()
            self.config = {
                "conf_thres": 0.4,
                "iou_thres": 0.3,
                "imgsz": 640,
                "tile_size": 320,
                "small_obj_size": 32,
                "alert_classes": [0, 2, 3, 5, 7],
                "alert_threshold": 3,
                "sharpening_kernel": np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]),
                "hist_bins": 64,
                "target_brightness": 0.4
            }

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.alert_active = False
        self.last_alert = 0

    def _auto_exposure_control(self, frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y = yuv[:,:,0]
        hist = cv2.calcHist([y], [0], None, [self.config["hist_bins"]], [0, 256])
        hist = hist / hist.sum()
        current_brightness = np.sum(hist * np.linspace(0, 1, self.config["hist_bins"]))
        error = self.config["target_brightness"] - current_brightness
        new_exposure = np.clip(1.0 + error * 0.5, 0.1, 1.5)
        try:
            cv2.setWindowProperty("Object Detection", cv2.WND_PROP_AUTOSIZE, new_exposure)
        except:
            pass

    def _tiled_detection(self, frame):
        h, w = frame.shape[:2]
        detections = []
        overlap = 64
        threads = []
        results = {}
        
        def process_tile(x, y, tile):
            tile = cv2.filter2D(tile, -1, self.config["sharpening_kernel"])
            results[(x, y)] = self.model.predict(
                tile,
                imgsz=self.config["imgsz"],
                conf=self.config["conf_thres"],
                iou=self.config["iou_thres"],
                device=self.device,
                verbose=False
            )

        for y in range(0, h, self.config["tile_size"] - overlap):
            for x in range(0, w, self.config["tile_size"] - overlap):
                tile = frame[y:y+self.config["tile_size"], x:x+self.config["tile_size"]]
                thread = threading.Thread(target=process_tile, args=(x, y, tile))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        for (x, y), result in results.items():
            for box in result[0].boxes:
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

    def _draw_metrics(self, frame, detections):
        self.frame_count += 1
        if time.time() - self.start_time > 1:
            self.fps = self.frame_count / (time.time() - self.start_time)
            self.frame_count = 0
            self.start_time = time.time()

        metrics = [
            f"FPS: {self.fps:.1f}",
            f"Objects: {len(detections)}",
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
        ]

        alert_count = sum(1 for d in detections if d["cls"] in self.config["alert_classes"])
        if alert_count >= self.config["alert_threshold"]:
            self.alert_active = True
            self.last_alert = time.time()
            cv2.putText(frame, "ALERT: High Activity!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if time.time() - self.last_alert < 2:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

        y_offset = 30
        for metric in metrics:
            cv2.putText(frame, metric, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30

        return frame

    def process_frame(self, frame):
        self._auto_exposure_control(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (self.config["imgsz"], self.config["imgsz"]))
        
        tensor_frame = torch.from_numpy(resized_frame).to(self.device).float() / 255.0
        tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            full_dets = self.model.predict(
                tensor_frame,
                imgsz=self.config["imgsz"],
                conf=self.config["conf_thres"],
                iou=self.config["iou_thres"],
                device=self.device,
                verbose=False
            )

        full_boxes = []
        scale_x = frame.shape[1] / self.config["imgsz"]
        scale_y = frame.shape[0] / self.config["imgsz"]
        
        for box in full_dets[0].boxes:
            adjusted_box = box.xyxy.clone()
            adjusted_box[0][0::2] *= scale_x
            adjusted_box[0][1::2] *= scale_y
            full_boxes.append({
                "xyxy": adjusted_box[0].cpu().numpy(),
                "conf": box.conf.item(),
                "cls": int(box.cls.item())
            })

        tile_boxes = self._tiled_detection(rgb_frame)
        combined_boxes = full_boxes + tile_boxes

        processed_frame = frame.copy()
        for box in combined_boxes:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        processed_frame = self._draw_metrics(processed_frame, combined_boxes)
        return processed_frame

def main():
    detector = SmallObjectDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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