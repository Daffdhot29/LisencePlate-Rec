import easyocr
import cv2
import re
from Utils.image_preprocessing import ImagePreprocessing
from Utils.plate_regex import PlateRegex


class OCRService:

    def __init__(self):

        # 🔥 pilih salah satu: CPU lebih stabil
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

        self.preprocessing = ImagePreprocessing()
        self.regex = PlateRegex()

    def recognize_plate(self, plate_crop):

        if plate_crop is None or plate_crop.size == 0:
            return ""

        variants = self.preprocessing.preprocess(plate_crop)

        best_text = ""

        for idx, variant in enumerate(variants):

            try:

                plate_resized = cv2.resize(
                    variant,
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

                ocr_result = self.reader.readtext(
                    plate_resized,
                    detail=0,
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )

                print(f"OCR Variant {idx}:", ocr_result)

                if len(ocr_result) == 0:
                    continue

                raw_text = ''.join(ocr_result)

                raw_text = re.sub(r'[^A-Z0-9]', '', raw_text)

                print("CLEAN TEXT:", raw_text)

                if len(raw_text) >= 5:
                    best_text = raw_text
                    break

            except Exception as e:
                print(f"OCR ERROR Variant {idx}: {e}")

        return best_text