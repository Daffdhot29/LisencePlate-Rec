import cv2


class ImagePreprocessing:

    def preprocess(
        self,
        plate_crop
    ):

        gray = cv2.cvtColor(
            plate_crop,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.bilateralFilter(
            gray,
            9,
            75,
            75
        )

        variants = []

        variants.append(gray)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

        clahe_image = clahe.apply(
            gray
        )

        variants.append(
            clahe_image
        )

        return variants