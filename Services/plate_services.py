import time
from collections import defaultdict, deque

import cv2
import easyocr
from ultralytics import YOLO

from app.utils.plate_utils import (
    get_box,
    format_plate,
    recognize_plate,
    draw_plate_text
)


class PlateRecognitionService:
    def __init__(self, model_path: str, gpu: bool = True):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(["en"], gpu=gpu)

        self.plate_history = defaultdict(lambda: deque(maxlen=15))
        self.plate_final = {}

        self.conf_thresh = 0.25
        self.pad = 12

    def get_stable_plate(self, box_id, new_text):
        if new_text:
            self.plate_history[box_id].append(new_text)

            most_common = max(
                set(self.plate_history[box_id]),
                key=self.plate_history[box_id].count
            )

            self.plate_final[box_id] = most_common

        return self.plate_final.get(box_id, "")

    def detect_image(self, input_path: str, output_path: str):
        start_time = time.time()

        img = cv2.imread(input_path)

        if img is None:
            raise ValueError("Input gambar tidak ditemukan")

        height, width = img.shape[:2]
        frame = img.copy()

        results = self.model(frame, verbose=False)

        detections = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf.item())

                if conf < self.conf_thresh:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                crop_x1 = max(0, x1 - self.pad)
                crop_y1 = max(0, y1 - self.pad)
                crop_x2 = min(width, x2 + self.pad)
                crop_y2 = min(height, y2 + self.pad)

                plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                text = recognize_plate(
                    plate_crop=plate_crop,
                    reader=self.reader
                )

                stable_id = get_box(x1, x2, y1, y2)
                final_text = self.get_stable_plate(stable_id, text)

                display_text = format_plate(final_text) if final_text else "PLATE DETECTED"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                draw_plate_text(
                    frame=frame,
                    display_text=display_text,
                    x1=x1,
                    y1=y1,
                    y2=y2
                )

                detections.append({
                    "plate_number": display_text if final_text else None,
                    "raw_text": final_text,
                    "yolo_confidence": round(conf, 2),
                    "box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        cv2.imwrite(output_path, frame)

        end_time = time.time()

        return {
            "detections": detections,
            "total_detected": len(detections),
            "process_time": round(end_time - start_time, 2),
            "saved_image": output_path
        }