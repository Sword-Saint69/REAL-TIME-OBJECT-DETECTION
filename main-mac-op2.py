import cv2
import torch
import time
import numpy as np
import os
from ultralytics import YOLO

os.environ['HF_HOME'] = os.path.expanduser('~/hf_cache')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

class EnhancedObjectDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8m.pt").to(self.device)
        
        self.config = {
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "input_size": 640,
            "tile_sizes": [320, 640],
            "small_obj_size": 64,
            "overlap": 128,
            "clahe_clip": 2.0,
            "sharpening": 1.7,
            "classes": [0, 1, 2, 3, 5, 7],
            "nms_thres": 0.5
        }

        if self.device == "cpu":
            torch.set_num_threads(torch.get_num_threads())
        else:
            self.model.fuse()
            self.model.half()

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def _enhance_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.config["clahe_clip"], tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced, (0,0), 3)
        sharpened = cv2.addWeighted(enhanced, self.config["sharpening"], blurred, 1 - self.config["sharpening"], 0)
        return sharpened

    def _multi_scale_detect(self, frame):
        detections = []
        
        # Full frame detection
        full_processed = self._enhance_image(frame)
        full_res = self.model.predict(
            full_processed,
            imgsz=self.config["input_size"],
            conf=self.config["conf_thres"],
            iou=self.config["iou_thres"],
            classes=self.config["classes"],
            device=self.device,
            verbose=False
        )
        detections.extend(full_res[0].boxes)
        
        # Multi-scale tiled detection
        for tile_size in self.config["tile_sizes"]:
            h, w = frame.shape[:2]
            for y in range(0, h, tile_size - self.config["overlap"]):
                for x in range(0, w, tile_size - self.config["overlap"]):
                    tile = frame[y:y+tile_size, x:x+tile_size]
                    enhanced_tile = self._enhance_image(tile)
                    
                    tile_res = self.model.predict(
                        enhanced_tile,
                        imgsz=tile_size,
                        conf=self.config["conf_thres"] * 0.9,
                        iou=self.config["iou_thres"],
                        classes=self.config["classes"],
                        device=self.device,
                        verbose=False
                    )
                    
                    for box in tile_res[0].boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        coords[0::2] += x
                        coords[1::2] += y
                        
                        # Create new box with adjusted coordinates
                        new_coords = torch.tensor([coords], device=self.device)
                        new_box = type(box)(
                            new_coords,
                            box.conf,
                            box.cls,
                            box.id,
                            orig_shape=frame.shape[:2]
                        )
                        detections.append(new_box)

        return detections

    def _non_max_suppression(self, boxes):
        dets = np.array([[b.xyxy[0][0].item(), b.xyxy[0][1].item(), 
                        b.xyxy[0][2].item(), b.xyxy[0][3].item(), 
                        b.conf.item(), b.cls.item()] for b in boxes])
        
        if len(dets) == 0:
            return []

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            inds = np.where(iou <= self.config["nms_thres"])[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def process_frame(self, frame):
        if frame is None:
            return None
            
        raw_detections = self._multi_scale_detect(frame)
        filtered_boxes = self._non_max_suppression(raw_detections)
        
        output = frame.copy()
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            
            width = x2 - x1
            height = y2 - y1
            
            color = (0, 0, 255) if max(width, height) < self.config["small_obj_size"] else (0, 255, 0)
            thickness = 2 if color == (0, 0, 255) else 1
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            label = f"{self.model.names[cls_id]} {conf:.2f}"
            cv2.putText(output, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        cv2.putText(output, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return output

def main():
    detector = EnhancedObjectDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame error, retrying...")
                time.sleep(0.1)
                cap.release()
                cap = cv2.VideoCapture(0)
                continue

            processed = detector.process_frame(frame)
            if processed is not None:
                cv2.imshow("Enhanced Detection", processed)

            if cv2.getWindowProperty("Enhanced Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()