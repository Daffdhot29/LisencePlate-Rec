import re
import threading
from typing import Any

import cv2
import numpy as np
import torch
 
from PIL import Image
from strhub.data.module import SceneTextDataModule


class PARSeqRecognizer:
    def __init__(
        self,
        model_name: str = "parseq",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self._lock = threading.Lock()

        self.model = torch.hub.load(
            "baudm/parseq",
            model_name,
            pretrained=True,
            trust_repo=True,
        )

        self.model = self.model.eval().to(
            self.device
        )

        self.transform = (
            SceneTextDataModule.get_transform(
                self.model.hparams.img_size
            )
        )

        print(
            f"PARSeq loaded: "
            f"{self.model_name} "
            f"on {self.device}"
        )

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).upper()

        return re.sub(
            r"[^A-Z0-9]",
            "",
            text,
        )

    @staticmethod
    def format_plate(text: str) -> str:
        cleaned = PARSeqRecognizer.clean_text(
            text
        )

        match = re.fullmatch(
            r"([A-Z]{1,2})"
            r"([0-9]{1,4})"
            r"([A-Z]{1,3})",
            cleaned,
        )

        if match is None:
            return cleaned

        return (
            f"{match.group(1)} "
            f"{match.group(2)} "
            f"{match.group(3)}"
        )

    @staticmethod
    def is_valid_plate(text: str) -> bool:
        cleaned = PARSeqRecognizer.clean_text(
            text
        )

        return (
            re.fullmatch(
                r"[A-Z]{1,2}"
                r"[0-9]{1,4}"
                r"[A-Z]{1,3}",
                cleaned,
            )
            is not None
        )

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> None:
        if image is None:
            raise ValueError(
                "Image PARSeq tidak boleh None"
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "Image PARSeq harus numpy.ndarray"
            )

        if image.size == 0:
            raise ValueError(
                "Crop plat kosong"
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                "Format image tidak valid"
            )

    @staticmethod
    def _opencv_to_pil(
        image: np.ndarray,
    ) -> Image.Image:
        PARSeqRecognizer._validate_image(
            image
        )

        if image.ndim == 2:
            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB,
            )
        elif image.shape[2] == 4:
            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGB,
            )
        else:
            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        return Image.fromarray(
            rgb_image
        )

    def _prepare_tensor(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        pil_image = self._opencv_to_pil(
            image
        ).convert("RGB")

        tensor = self.transform(
            pil_image
        )

        tensor = tensor.unsqueeze(0)

        return tensor.to(
            self.device,
            non_blocking=(
                self.device == "cuda"
            ),
        )

    @staticmethod
    def _extract_confidence(
        confidence: Any,
    ) -> float:
        if confidence is None:
            return 0.0

        if torch.is_tensor(confidence):
            if confidence.numel() == 0:
                return 0.0

            values = confidence.detach().float()

            if values.ndim == 0:
                return float(values.item())

            first_item = values[0]

            if first_item.ndim == 0:
                return float(
                    first_item.item()
                )

            valid_values = first_item[
                first_item > 0
            ]

            if valid_values.numel() == 0:
                return 0.0

            return float(
                valid_values.mean().item()
            )

        if isinstance(
            confidence,
            (
                list,
                tuple,
                np.ndarray,
            ),
        ):
            values = np.asarray(
                confidence,
                dtype=np.float32,
            ).reshape(-1)

            values = values[
                values > 0
            ]

            if values.size == 0:
                return 0.0

            return float(
                values.mean()
            )

        try:
            return float(confidence)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    def recognize(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        try:
            input_tensor = (
                self._prepare_tensor(
                    image
                )
            )

            with self._lock:
                with torch.inference_mode():
                    logits = self.model(
                        input_tensor
                    )

                    probabilities = (
                        logits.softmax(-1)
                    )

                    labels, confidences = (
                        self.model.tokenizer.decode(
                            probabilities
                        )
                    )

            raw_text = (
                labels[0]
                if labels
                else ""
            )

            cleaned_text = self.clean_text(
                raw_text
            )

            confidence = (
                self._extract_confidence(
                    confidences
                )
            )

            return {
                "raw_text": str(raw_text),
                "text": cleaned_text,
                "plate": self.format_plate(
                    cleaned_text
                ),
                "confidence": round(
                    confidence,
                    4,
                ),
                "valid_format": (
                    self.is_valid_plate(
                        cleaned_text
                    )
                ),
                "model": self.model_name,
                "device": self.device,
                "error": None,
            }

        except Exception as error:
            print(
                "PARSEQ ERROR:",
                error,
            )

            return {
                "raw_text": "",
                "text": "",
                "plate": "",
                "confidence": 0.0,
                "valid_format": False,
                "model": self.model_name,
                "device": self.device,
                "error": str(error),
            }

    def recognize_simple(
        self,
        image: np.ndarray,
    ) -> tuple[str, float]:
        result = self.recognize(
            image
        )

        return (
            result["text"],
            float(
                result["confidence"]
            ),
        )