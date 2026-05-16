from ultralytics import YOLO
from Services.ocr_service import OCRService
from Utils.plate_history import PlateHistory
import cv2
import os


class YOLOService:

    def __init__(self):

        model_path = os.path.abspath(
            "models/license_plate_M.pt"
        )

        self.model = YOLO(model_path)

        self.ocr_service = OCRService()
        self.plate_history = PlateHistory()

        self.conf_thresh = 0.25  

    def detect_plate(self, frame):

        height, width = frame.shape[:2]

        results = self.model(frame, verbose=False)

        plate_candidates = {}

        for r in results:
            for box in r.boxes:

                yolo_conf = float(box.conf.item())

                if yolo_conf < self.conf_thresh:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy.cpu().numpy()[0]
                )

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                pad = 5

                plate_crop = frame[
                    max(0, y1 - pad):min(height, y2 + pad),
                    max(0, x1 - pad):min(width, x2 + pad)
                ]

                if plate_crop.size == 0:
                    continue

             
                text = self.ocr_service.recognize_plate(plate_crop)

                if not text:
                    continue

        
                box_id = self.plate_history.get_box(
                    x1, x2, y1, y2
                )

                # simpan history OCR (bukan YOLO conf)
                stable_text = self.plate_history.get_stable_plate(
                    box_id,
                    text
                )

                if not stable_text:
                    continue

          
                if box_id not in plate_candidates:
                    plate_candidates[box_id] = {
                        "plate_number": stable_text,
                        "yolo_conf": round(yolo_conf, 2)
                    }

                else:
                    if len(stable_text) > len(plate_candidates[box_id]["plate_number"]):
                        plate_candidates[box_id] = {
                            "plate_number": stable_text,
                            "yolo_conf": round(yolo_conf, 2)
                        }

        return list(plate_candidates.values())