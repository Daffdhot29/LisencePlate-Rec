import cv2
import numpy as np


class ImagePreprocessing:

    def preprocess(self, plate_crop):

        if plate_crop is None or plate_crop.size == 0:
            return []

        variants = []

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        variants.append(gray)

    
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        variants.append(denoised)

      
        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )
        clahe_img = clahe.apply(gray)
        variants.append(clahe_img)

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        variants.append(adaptive)

      
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        sharpen = cv2.filter2D(gray, -1, kernel)
        variants.append(sharpen)

  
        enhanced = []

        for v in variants:
            resized = cv2.resize(
                v,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )
            enhanced.append(resized)

        return enhanced