from ultralytics import YOLO
import cv2
import cvzone
import math
import os

class YOLO_model:
    def __init__(self, model_path="yolov10m.pt"):   # Constructor
        self.yolo_instance = YOLO(model_path, verbose=False)

    def process_media(self, file_path):
        # Check file extension to ensure it's a video or image
        _, file_extension = os.path.splitext(file_path)
        if file_extension not in [".mp4", ".mov", ".jpeg", ".jpg", ".png"]:
            raise ValueError("Unsupported file format")

        # Capture video
        media = cv2.VideoCapture(file_path)
        processed_frames = []
        annotations = []  # List to store annotations for each frame
        
        while True:
            success, img = media.read()
            if not success:
                break  # Break loop if no frame is returned

            results = self.yolo_instance(img, stream=True)

            frame_annotations = []  # List to store annotations for the current frame

            for r in results:
                boxes = r.boxes

                for box in boxes: 
                    # Get bounding box and class information
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h  = x2 - x1, y2 - y1
                    bounding_box = (x1, y1, w, h)
                    cvzone.cornerRect(img, bounding_box)

                    confidence = math.floor((box.conf[0] * 100)) / 100  
                    cls_id = int(box.cls[0])
                    obj_name = self.yolo_instance.names[cls_id]

                    # Adds box to frame
                    obj_box = (max(0, x1), max(35, y1))
                    cvzone.putTextRect(img, f"{obj_name} {confidence}", obj_box, scale=1, thickness=2)

                    # Store annotation data
                    annotation = {
                        "class": obj_name,
                        "confidence": confidence,
                        "bounding_box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": w,
                            "height": h
                        }
                    }
                    frame_annotations.append(annotation)

            annotations.append(frame_annotations)  # Add frame's annotations to the list
            processed_frames.append(img)  # Store processed frame

        media.release()

        return processed_frames, annotations
