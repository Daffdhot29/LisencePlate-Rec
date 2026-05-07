import easyocr
import cv2
import re

from Utils.image_preprocessing import (
    ImagePreprocessing
)

from Utils.plate_regex import (
    PlateRegex
)


class OCRService:

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en'],
            gpu=True
        )

        self.preprocessing = ImagePreprocessing()

        self.regex = PlateRegex()

    def recognize_plate(
        self,
        plate_crop
    ):

        if plate_crop.size == 0:
            return ""

        variants = self.preprocessing.preprocess(
            plate_crop
        )

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

                print(
                    f"OCR Variant {idx}:",
                    ocr_result
                )

                if len(ocr_result) >= 3:

                    raw_text = ''.join(
                        ocr_result[:3]
                    )

                    raw_text = re.sub(
                        r'[^A-Z0-9]',
                        '',
                        raw_text
                    )

                    print(
                        "CLEAN TEXT:",
                        raw_text
                    )

                    if self.regex.validate(
                        raw_text
                    ):

                        print(
                            "VALID PLATE:",
                            raw_text
                        )

                        return raw_text

            except Exception as e:

                print(
                    f"OCR ERROR Variant {idx}: {e}"
                )

        return ""