import easyocr
import cv2
import re
from Utils.image_preprocessing import ImagePreprocessing
from Utils.plate_regex import PlateRegex


class OCRService:

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

        self.preprocessing = ImagePreprocessing()
        self.regex = PlateRegex()

    # =========================
    # CLEAN TEXT
    # =========================
    def _clean_text(self, text):
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text

    # =========================
    # MAIN OCR FUNCTION
    # =========================
    def recognize_plate(self, plate_crop):

        if plate_crop is None or plate_crop.size == 0:
            return ""

        variants = self.preprocessing.preprocess(plate_crop)

        best_text = ""
        best_score = 0.0

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
                    detail=1,
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )

                if not ocr_result:
                    continue

           
                ocr_result = sorted(
                    ocr_result,
                    key=lambda x: x[0][0][0]
                )

           
                raw_text = ""
                total_conf = 0.0

                for (_, text, conf) in ocr_result:
                    text = self._clean_text(text)

                    if not text:
                        continue

                    raw_text += text
                    total_conf += float(conf)

                if len(raw_text) < 4:
                    continue

                avg_conf = total_conf / max(len(ocr_result), 1)

                
                structure_bonus = 0.3 if self.regex.validate(raw_text) else 0.0

                # length stability bonus
                length_bonus = min(len(raw_text) / 8.0, 1.0) * 0.1

             
                score = (avg_conf * 0.6) + structure_bonus + length_bonus

                print(f"OCR Variant {idx}: {raw_text} | SCORE: {score}")

  
                if score > best_score:
                    best_score = score
                    best_text = raw_text

            except Exception as e:
                print(f"OCR ERROR Variant {idx}: {e}")

        return best_text