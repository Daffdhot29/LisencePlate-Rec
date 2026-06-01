from fastapi import APIRouter, UploadFile, File
from ultralytics import YOLO
from collections import defaultdict, deque
import easyocr
import cv2
import re
import shutil
import uuid
import os
import time


class PlateRecognizer:

    def __init__(self):
        self.router = APIRouter(
            prefix="/plate",
            tags=["License Plate"]
        )

        self.router.add_api_route(
            "/detect",
            self.detect_plate_api,
            methods=["POST"]
        )

        self.model = YOLO("models/license_plate_bestM.pt")
        self.reader = easyocr.Reader(["en"], gpu=True)

        self.plate_pattern = re.compile(
            r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$"
        )

        self.plate_history = defaultdict(lambda: deque(maxlen=15))
        self.plate_final = {}

        self.CONF_THRESH = 0.25
        self.PAD = 12

    async def detect_plate_api(self, file: UploadFile = File(...)):
        start_time = time.time()

        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

        input_path = f"uploads/{uuid.uuid4()}_{file.filename}"
        output_path = f"outputs/result_{file.filename}"

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        detections = self.process_image(input_path, output_path)

        end_time = time.time()

        return {
            "success": True,
            "detections": detections,
            "total_detected": len(detections),
            "process_time": round(end_time - start_time, 2),
            "output_file": output_path
        }

    def get_box(self, x1, x2, y1, y2):
        return f"{int(x1//15)}_{int(x2//15)}_{int(y1//15)}_{int(y2//15)}"

    def get_stable_plate(self, box_id, new_text):
        if new_text:
            self.plate_history[box_id].append(new_text)

            most_common = max(
                set(self.plate_history[box_id]),
                key=self.plate_history[box_id].count
            )

            self.plate_final[box_id] = most_common

        return self.plate_final.get(box_id, "")

    def format_plate(self, text):
        match = re.match(
            r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})$",
            text
        )

        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)}"

        return text

    def ocr_with_confidence(self, image):
        try:
            result = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            output = []

            for _, text, conf in result:
                text = re.sub(r"[^A-Z0-9]", "", text.upper())

                if text:
                    output.append((text, conf))

            return output

        except Exception as e:
            print("OCR ERROR:", e)
            return []

    def preprocess_variants(self, plate_crop):
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        clahe_img = clahe.apply(blur)

        _, otsu = cv2.threshold(
            clahe_img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        inverted = cv2.bitwise_not(otsu)

        return [gray, clahe_img, otsu, inverted]

    def recognize_plate(self, plate_crop):
        if plate_crop is None or plate_crop.size == 0:
            return ""

        variants = self.preprocess_variants(plate_crop)

        best_text = ""
        best_score = 0

        for variant in variants:
            ocr_results = self.ocr_with_confidence(variant)

            if not ocr_results:
                continue

            joined_text = "".join([text for text, _ in ocr_results])
            joined_text = re.sub(r"[^A-Z0-9]", "", joined_text.upper())

            avg_conf = sum(conf for _, conf in ocr_results) / len(ocr_results)

            if len(joined_text) >= 4:
                regex_bonus = 1.0 if self.plate_pattern.match(joined_text) else 0.6
                score = avg_conf * regex_bonus

                if score > best_score:
                    best_score = score
                    best_text = joined_text

        return best_text

    def process_image(self, input_path, output_path):
        img = cv2.imread(input_path)

        if img is None:
            raise ValueError("Input gambar tidak ditemukan")

        frame = img.copy()
        height, width = frame.shape[:2]

        results = self.model(frame, verbose=False)

        detections = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf.item())

                if conf < self.CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy.cpu().numpy()[0]
                )

                crop_x1 = max(0, x1 - self.PAD)
                crop_y1 = max(0, y1 - self.PAD)
                crop_x2 = min(width, x2 + self.PAD)
                crop_y2 = min(height, y2 + self.PAD)

                plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                text = self.recognize_plate(plate_crop)

                stable_id = self.get_box(x1, x2, y1, y2)
                final_text = self.get_stable_plate(stable_id, text)

                if final_text:
                    final_text = self.format_plate(final_text)

                    detections.append({
                        "plate": final_text,
                        "confidence": round(conf, 2)
                    })

        cv2.imwrite(output_path, frame)

        return detections


plate_recognizer = PlateRecognizer()
router = plate_recognizer.router