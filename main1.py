import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Configure environment
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

class RobustObjectDetector:
    def __init__(self):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize YOLOv8 detector
        self.detector = YOLO("yolov8n.pt").to(self.device)
        
        # Initialize super-resolution model
        self.sr_model = self._init_super_resolution().to(self.device)
        self.sr_model.eval()

    def _init_super_resolution(self):
        """Create a simple upsampling model"""
        return torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        )

    def _enhance_frame(self, frame):
        """Enhanced frame processing with contiguous array conversion"""
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(frame).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Apply super-resolution
        with torch.no_grad():
            enhanced = self.sr_model(img_tensor)
        
        # Convert back to numpy array and ensure contiguous
        enhanced = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray((np.clip(enhanced, 0, 1) * 255).astype(np.uint8))

    def process_frame(self, frame):
        """Complete processing pipeline with error handling"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Enhance frame
            enhanced = self._enhance_frame(rgb_frame)
            
            # Detect objects
            results = self.detector.predict(
                enhanced,
                conf=0.6,
                iou=0.5,
                classes=[0, 2, 3, 5, 7],
                imgsz=640,
                device=self.device
            )
            
            # Return annotated frame with contiguous array
            return results[0].plot()
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return frame

def main():
    detector = RobustObjectDetector()
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = detector.process_frame(frame)
            cv2.imshow("Robust Detection", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()