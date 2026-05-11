from ultralytics import YOLO
from Services.ocr_service import OCRService
from Utils.plate_history import PlateHistory
import cv2


class YOLOService:

    def __init__(self):

        self.model = YOLO(
            "../models/license_plate_best_new.pt"
        )

        self.ocr_service = OCRService()

        self.plate_history = PlateHistory()

        self.conf_thresh = 0.6

    def detect_plate(
        self,
        frame
    ):

        height, width = frame.shape[:2]

        detected_plates = []

        results = self.model(
            frame,
            verbose=False,
            device=0
        )

        for r in results:

            boxes = r.boxes

            for box in boxes:

                conf = float(
                    box.conf.item()
                )

                if conf < self.conf_thresh:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy.cpu().numpy()[0]
                )

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                pad = 2

                plate_crop = frame[
                    max(0, y1-pad):min(height, y2+pad),
                    max(0, x1-pad):min(width, x2+pad)
                ]

                text = self.ocr_service.recognize_plate(
                    plate_crop
                )

                box_id = self.plate_history.get_box(
                    x1,
                    x2,
                    y1,
                    y2
                )

                stable_text = self.plate_history.get_stable_plate(
                    box_id,
                    text
                )

                if stable_text:

                    detected_plates.append({
                        "plate_number": stable_text,
                        "confidence": round(conf, 2),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    })

        return detected_plates