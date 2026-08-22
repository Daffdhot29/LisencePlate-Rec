import os
import re
import time
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR


app = FastAPI(
    title="LPR API"
)


MODEL_PATH = "models/model_lpr3.onnx"

IMG_SIZE = 640

PLATE_CONF_THRESH = 0.20
VEHICLE_CONF_THRESH = 0.25
IOU_THRESH = 0.45

OCR_MIN_CONFIDENCE = 0.20

LICENSE_PLATE_CLASS_ID = 0

CLASS_NAMES = [
    "License_Plate",
    "cars",
    "motorcycle",
]

PLATE_PATTERN = re.compile(
    r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$"
)


def clean_text(text: str) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )


def is_valid_plate(text: str) -> bool:
    return (
        PLATE_PATTERN.fullmatch(
            clean_text(text)
        )
        is not None
    )


def format_plate(text: str) -> str:
    cleaned = clean_text(text)

    match = re.fullmatch(
        r"([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})",
        cleaned,
    )

    if match is None:
        return cleaned

    return (
        f"{match.group(1)} "
        f"{match.group(2)} "
        f"{match.group(3)}"
    )


def get_onnx_providers() -> list[str]:
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        return [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model ONNX tidak ditemukan: {MODEL_PATH}"
    )


detector_session = ort.InferenceSession(
    MODEL_PATH,
    providers=get_onnx_providers(),
)

detector_input_name = (
    detector_session.get_inputs()[0].name
)

detector_input_shape = (
    detector_session.get_inputs()[0].shape
)

detector_output_shape = (
    detector_session.get_outputs()[0].shape
)


paddle_ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    show_log=False,
)

paddle_lock = threading.Lock()


def make_ocr_variants(
    crop: np.ndarray,
) -> list[tuple[str, np.ndarray]]:

    if crop is None or crop.size == 0:
        return []

    height = crop.shape[0]

    target_height = 128

    scale = max(
        2.0,
        min(
            5.0,
            target_height / max(height, 1),
        ),
    )

    resized_color = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(
        resized_color,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8),
    )

    clahe_image = clahe.apply(gray)

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    sharpen_image = cv2.filter2D(
        clahe_image,
        -1,
        sharpen_kernel,
    )

    _, otsu_image = cv2.threshold(
        clahe_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return [
        (
            "original",
            resized_color,
        ),
        (
            "gray",
            cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR,
            ),
        ),
        (
            "clahe",
            cv2.cvtColor(
                clahe_image,
                cv2.COLOR_GRAY2BGR,
            ),
        ),
        (
            "sharpen",
            cv2.cvtColor(
                sharpen_image,
                cv2.COLOR_GRAY2BGR,
            ),
        ),
        (
            "otsu",
            cv2.cvtColor(
                otsu_image,
                cv2.COLOR_GRAY2BGR,
            ),
        ),
    ]


def run_paddle_ocr(
    image: np.ndarray,
) -> list[tuple[str, float]]:

    try:
        with paddle_lock:
            result = paddle_ocr.ocr(
                image,
                cls=False,
            )

        if not result or not result[0]:
            return []

        outputs: list[tuple[str, float]] = []

        for line in result[0]:
            if not line or len(line) < 2:
                continue

            raw_text = str(line[1][0])
            confidence = float(line[1][1])

            cleaned = clean_text(raw_text)

            if not cleaned:
                continue

            if confidence < OCR_MIN_CONFIDENCE:
                continue

            outputs.append(
                (
                    cleaned,
                    confidence,
                )
            )

        return outputs

    except Exception as error:
        print(
            "PADDLE OCR ERROR:",
            error,
        )

        return []


def score_ocr_result(
    text: str,
    confidence: float,
) -> float:

    cleaned = clean_text(text)

    score = float(confidence)

    if is_valid_plate(cleaned):
        score += 1.0

    if 6 <= len(cleaned) <= 9:
        score += 0.10

    if len(cleaned) < 5:
        score -= 0.50

    if len(cleaned) > 10:
        score -= 0.50

    return score


def recognize_plate(
    crop: np.ndarray,
) -> dict[str, Any]:

    if crop is None or crop.size == 0:
        return {
            "text": "",
            "plate": "",
            "confidence": 0.0,
        }

    best_text = ""
    best_confidence = 0.0
    best_score = float("-inf")

    variants = make_ocr_variants(crop)

    for _, variant in variants:

        ocr_results = run_paddle_ocr(
            variant
        )

        if not ocr_results:
            continue

        results_to_check = list(
            ocr_results
        )

        if len(ocr_results) > 1:

            joined_text = clean_text(
                "".join(
                    text
                    for text, _
                    in ocr_results
                )
            )

            average_confidence = (
                sum(
                    confidence
                    for _, confidence
                    in ocr_results
                )
                / len(ocr_results)
            )

            results_to_check.append(
                (
                    joined_text,
                    average_confidence,
                )
            )

        for text, confidence in results_to_check:

            cleaned = clean_text(text)

            if not cleaned:
                continue

            current_score = score_ocr_result(
                cleaned,
                confidence,
            )

            if current_score > best_score:
                best_text = cleaned
                best_confidence = confidence
                best_score = current_score

    if not best_text:
        return {
            "text": "",
            "plate": "",
            "confidence": 0.0,
        }

    return {
        "text": best_text,
        "plate": format_plate(best_text),
        "confidence": round(
            float(best_confidence),
            4,
        ),
    }


def letterbox(
    image: np.ndarray,
    new_shape: int = IMG_SIZE,
) -> tuple[np.ndarray, float, int, int]:

    image_height, image_width = image.shape[:2]

    scale = min(
        new_shape / image_height,
        new_shape / image_width,
    )

    resized_width = int(
        round(image_width * scale)
    )

    resized_height = int(
        round(image_height * scale)
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (
            new_shape,
            new_shape,
            3,
        ),
        114,
        dtype=np.uint8,
    )

    pad_x = (
        new_shape - resized_width
    ) // 2

    pad_y = (
        new_shape - resized_height
    ) // 2

    canvas[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width,
    ] = resized

    return (
        canvas,
        scale,
        pad_x,
        pad_y,
    )


def preprocess_detector(
    image: np.ndarray,
) -> tuple[np.ndarray, float, int, int]:

    (
        detector_image,
        scale,
        pad_x,
        pad_y,
    ) = letterbox(
        image,
        IMG_SIZE,
    )

    detector_image = cv2.cvtColor(
        detector_image,
        cv2.COLOR_BGR2RGB,
    )

    detector_image = (
        detector_image.astype(np.float32)
        / 255.0
    )

    detector_image = np.transpose(
        detector_image,
        (2, 0, 1),
    )

    detector_image = np.expand_dims(
        detector_image,
        axis=0,
    )

    detector_image = np.ascontiguousarray(
        detector_image
    )

    return (
        detector_image,
        scale,
        pad_x,
        pad_y,
    )


def xywh_to_xyxy(
    box: np.ndarray,
) -> tuple[float, float, float, float]:

    center_x = float(box[0])
    center_y = float(box[1])
    width = float(box[2])
    height = float(box[3])

    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def get_class_threshold(
    class_id: int,
) -> float:

    if class_id == LICENSE_PLATE_CLASS_ID:
        return PLATE_CONF_THRESH

    return VEHICLE_CONF_THRESH


def normalize_predictions(
    output: np.ndarray,
) -> np.ndarray:

    predictions = output

    if predictions.ndim == 3:
        predictions = predictions[0]

    expected_columns = (
        4 + len(CLASS_NAMES)
    )

    if predictions.shape[0] == expected_columns:
        predictions = predictions.T

    elif predictions.shape[1] == expected_columns:
        pass

    elif predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    return predictions


def postprocess(
    outputs: list[np.ndarray],
    original_image: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> list[dict[str, Any]]:

    predictions = normalize_predictions(
        outputs[0]
    )

    image_height, image_width = (
        original_image.shape[:2]
    )

    boxes: list[list[int]] = []
    scores: list[float] = []
    class_ids: list[int] = []

    expected_columns = (
        4 + len(CLASS_NAMES)
    )

    for prediction in predictions:

        if prediction.shape[0] < expected_columns:
            continue

        class_scores = prediction[
            4:4 + len(CLASS_NAMES)
        ]

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )

        threshold = get_class_threshold(
            class_id
        )

        if confidence < threshold:
            continue

        x1, y1, x2, y2 = xywh_to_xyxy(
            prediction[:4]
        )

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = int(
            np.clip(
                round(x1),
                0,
                image_width - 1,
            )
        )

        y1 = int(
            np.clip(
                round(y1),
                0,
                image_height - 1,
            )
        )

        x2 = int(
            np.clip(
                round(x2),
                0,
                image_width,
            )
        )

        y2 = int(
            np.clip(
                round(y2),
                0,
                image_height,
            )
        )

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append(
            [
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            ]
        )

        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    detections = []

    for class_id in sorted(set(class_ids)):

        class_indices = [
            index
            for index, current_class_id
            in enumerate(class_ids)
            if current_class_id == class_id
        ]

        class_boxes = [
            boxes[index]
            for index in class_indices
        ]

        class_scores = [
            scores[index]
            for index in class_indices
        ]

        nms_indices = cv2.dnn.NMSBoxes(
            class_boxes,
            class_scores,
            get_class_threshold(class_id),
            IOU_THRESH,
        )

        if len(nms_indices) == 0:
            continue

        for local_index in (
            np.array(nms_indices).reshape(-1)
        ):

            global_index = class_indices[
                int(local_index)
            ]

            x, y, width, height = (
                boxes[global_index]
            )

            detections.append(
                {
                    "class_id": class_ids[
                        global_index
                    ],
                    "class_name": CLASS_NAMES[
                        class_ids[
                            global_index
                        ]
                    ],
                    "det_confidence": float(
                        scores[
                            global_index
                        ]
                    ),
                    "box": [
                        x,
                        y,
                        x + width,
                        y + height,
                    ],
                }
            )

    return detections


def get_vehicle_type(
    detections: list[dict[str, Any]],
) -> str:

    vehicle_detections = [
        detection
        for detection in detections
        if detection["class_name"]
        in {
            "cars",
            "motorcycle",
        }
    ]

    if not vehicle_detections:
        return "Unknown"

    best_vehicle = max(
        vehicle_detections,
        key=lambda detection:
            detection["det_confidence"],
    )

    return str(
        best_vehicle["class_name"]
    )


def crop_plate_for_ocr(
    image: np.ndarray,
    box: list[int],
) -> np.ndarray:

    image_height, image_width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = [
        int(value)
        for value in box
    ]

    box_width = max(
        1,
        x2 - x1,
    )

    box_height = max(
        1,
        y2 - y1,
    )

    padding_x = int(
        box_width * 0.12
    )

    padding_top = int(
        box_height * 0.10
    )

    padding_bottom = int(
        box_height * 0.05
    )

    padding_x = max(
        3,
        min(
            padding_x,
            24,
        ),
    )

    padding_top = max(
        2,
        min(
            padding_top,
            14,
        ),
    )

    padding_bottom = max(
        1,
        min(
            padding_bottom,
            8,
        ),
    )

    crop_x1 = max(
        0,
        x1 - padding_x,
    )

    crop_y1 = max(
        0,
        y1 - padding_top,
    )

    crop_x2 = min(
        image_width,
        x2 + padding_x,
    )

    crop_y2 = min(
        image_height,
        y2 + padding_bottom,
    )

    return image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]


def select_best_plate_detection(
    detections: list[dict[str, Any]],
) -> dict[str, Any] | None:

    plate_detections = [
        detection
        for detection in detections
        if detection["class_id"]
        == LICENSE_PLATE_CLASS_ID
    ]

    if not plate_detections:
        return None

    return max(
        plate_detections,
        key=lambda detection:
            detection["det_confidence"],
    )


def process_image(
    image: np.ndarray,
) -> dict[str, Any]:

    total_start_time = time.time()

    (
        detector_tensor,
        scale,
        pad_x,
        pad_y,
    ) = preprocess_detector(image)

    outputs = detector_session.run(
        None,
        {
            detector_input_name:
                detector_tensor
        },
    )

    detections = postprocess(
        outputs,
        image,
        scale,
        pad_x,
        pad_y,
    )

    vehicle_type = get_vehicle_type(
        detections
    )

    plate_detection = (
        select_best_plate_detection(
            detections
        )
    )

    if plate_detection is None:
        return {
            "status": "plate_not_detected",
            "vehicle_type": vehicle_type,
            "plate": "Plate Unreadable",
            "raw_plate": "",
            "processing_time_seconds": round(
                time.time() - total_start_time,
                4,
            ),
        }

    plate_crop = crop_plate_for_ocr(
        image,
        plate_detection["box"],
    )

    ocr_result = recognize_plate(
        plate_crop
    )

    plate_text = (
        ocr_result["plate"]
        if ocr_result["text"]
        else "Plate Unreadable"
    )

    return {
        "status": "success",
        "vehicle_type": vehicle_type,
        "plate": plate_text,
        "raw_plate": ocr_result["text"],
        "processing_time_seconds": round(
            time.time() - total_start_time,
            4,
        ),
    }


def invalid_image_response(
    error: str,
) -> dict[str, Any]:

    return {
        "status": "error",
        "vehicle_type": "Unknown",
        "plate": "Plate Unreadable",
        "raw_plate": "",
        "processing_time_seconds": 0.0,
        "error": error,
    }


@app.get("/")
def root() -> dict[str, Any]:

    return {
        "message": "ALPR API is running",
        "endpoint": "POST /recognize",
        "detector": "YOLOv9-tiny ONNX",
        "model": MODEL_PATH,
        "img_size": IMG_SIZE,
        "ocr": "PaddleOCR",
    }


@app.post("/recognize")
async def recognize(
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
):

    try:
        uploaded_file = file or image

        if uploaded_file is None:
            return JSONResponse(
                status_code=400,
                content=invalid_image_response(
                    'File gambar tidak ditemukan. '
                    'Gunakan multipart/form-data '
                    'dengan field "file" atau "image".'
                ),
            )

        file_bytes = await uploaded_file.read()

        if not file_bytes:
            return JSONResponse(
                status_code=400,
                content=invalid_image_response(
                    "File kosong"
                ),
            )

        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8,
        )

        decoded_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if decoded_image is None:
            return JSONResponse(
                status_code=400,
                content=invalid_image_response(
                    "File bukan gambar valid"
                ),
            )

        result = process_image(
            decoded_image
        )

        return JSONResponse(
            status_code=200,
            content=result,
        )

    except Exception as error:

        print(
            "API ERROR:",
            error,
        )

        return JSONResponse(
            status_code=500,
            content=invalid_image_response(
                str(error)
            ),
        )