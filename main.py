import os
import re
import cv2
import time
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR


app = FastAPI(title="ALPR API")

MODEL_PATH = "models/vehicle_plate_yolov9tiny_best.onnx"

IMG_SIZE = 960
CONF_THRESH = 0.35
IOU_THRESH = 0.45

CLASS_NAMES = [
    "License_Plate",
    "cars",
    "motorcyle",
    "truck"
]

LICENSE_PLATE_CLASS_ID = 0

plate_pattern = re.compile(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$")
plate_search_pattern = re.compile(r"[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}")


ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    show_log=False
)


def get_onnx_providers():
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


session = ort.InferenceSession(
    MODEL_PATH,
    providers=get_onnx_providers()
)

input_name = session.get_inputs()[0].name


def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def format_plate(text):
    text = clean_text(text)

    match = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})$", text)

    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"

    return text


def generate_plate_candidates(text):
    text = clean_text(text)

    if len(text) > 10:
        return [text]

    substitutions = {
        "0": ["0", "O"],
        "O": ["O", "0"],
        "1": ["1", "I", "L"],
        "I": ["I", "1"],
        "L": ["L", "1"],
        "8": ["8", "B"],
        "B": ["B", "8"],
        "5": ["5", "S"],
        "S": ["S", "5"],
        "6": ["6", "G"],
        "G": ["G", "6"],
        "2": ["2", "Z"],
        "Z": ["Z", "2"],
    }

    candidates = set()

    def backtrack(index, current):
        if index == len(text):
            candidates.add("".join(current))
            return

        char = text[index]
        options = substitutions.get(char, [char])

        for opt in options:
            current.append(opt)
            backtrack(index + 1, current)
            current.pop()

    backtrack(0, [])

    return list(candidates)


def fix_plate_by_position(text):
    text = clean_text(text)

    if len(text) < 4:
        return text

    to_digit = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8"
    }

    to_letter = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B"
    }

    chars = list(text)

    for i in range(len(chars)):
        if 1 <= i <= 5:
            chars[i] = to_digit.get(chars[i], chars[i])

    for i in range(len(chars)):
        if i >= 5:
            chars[i] = to_letter.get(chars[i], chars[i])

    return "".join(chars)


def extract_indonesian_plate(text):
    text = clean_text(text)

    candidates = plate_search_pattern.findall(text)

    if not candidates:
        return text

    candidates = sorted(
        candidates,
        key=lambda x: (
            plate_pattern.fullmatch(x) is not None,
            6 <= len(x) <= 9,
            len(x)
        ),
        reverse=True
    )

    return candidates[0]


def ocr_based_score(text, ocr_confidence):
    text = clean_text(text)

    score = ocr_confidence * 100

    if plate_pattern.fullmatch(text):
        score += 20

    if re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})$", text):
        score += 15

    if 6 <= len(text) <= 9:
        score += 5

    if len(text) < 5 or len(text) > 10:
        score -= 30

    return score


def preprocess_variants_for_ocr(crop):
    variants = []

    if crop is None or crop.size == 0:
        return variants

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    if w < 100:
        scale = 6
    elif w < 160:
        scale = 5
    elif w < 240:
        scale = 4
    else:
        scale = 3

    resized = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(resized)

    gaussian = cv2.GaussianBlur(clahe_img, (3, 3), 0)
    median = cv2.medianBlur(clahe_img, 3)

    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  5, -1],
        [-1, -1, -1]
    ])

    sharp = cv2.filter2D(clahe_img, -1, sharpen_kernel)

    _, otsu = cv2.threshold(
        clahe_img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        clahe_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    inverted_otsu = cv2.bitwise_not(otsu)
    inverted_adaptive = cv2.bitwise_not(adaptive)

    variants.append(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(gaussian, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(median, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(inverted_otsu, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(inverted_adaptive, cv2.COLOR_GRAY2BGR))

    return variants


def run_ocr(image):
    try:
        result = ocr.ocr(image, cls=False)

        if not result or not result[0]:
            return []

        outputs = []

        for line in result[0]:
            text = clean_text(line[1][0])
            conf = float(line[1][1])

            if text:
                outputs.append((text, conf))

        return outputs

    except Exception as e:
        print("OCR ERROR:", e)
        return []


def recognize_plate(crop):
    if crop is None or crop.size == 0:
        return "", 0.0, 0.0, []

    variants = preprocess_variants_for_ocr(crop)

    all_candidates = []

    for variant in variants:
        ocr_results = run_ocr(variant)

        if not ocr_results:
            continue

        joined_text = clean_text("".join([text for text, conf in ocr_results]))
        avg_conf = sum([conf for text, conf in ocr_results]) / len(ocr_results)

        raw_texts = [(joined_text, avg_conf)] + ocr_results

        for raw_text, conf in raw_texts:
            raw_text = clean_text(raw_text)

            if not raw_text:
                continue

            generated_candidates = generate_plate_candidates(raw_text)

            for candidate in generated_candidates:
                fixed = fix_plate_by_position(candidate)
                extracted = extract_indonesian_plate(fixed)
                final_candidate = clean_text(extracted)

                final_score = ocr_based_score(final_candidate, conf)

                all_candidates.append({
                    "value": final_candidate,
                    "formatted": format_plate(final_candidate),
                    "score": round(final_score, 3),
                    "ocr_confidence": round(conf, 3)
                })

    if not all_candidates:
        return "", 0.0, 0.0, []

    all_candidates = sorted(
        all_candidates,
        key=lambda x: (
            x["ocr_confidence"],
            x["score"]
        ),
        reverse=True
    )

    unique_candidates = []
    seen = set()

    for item in all_candidates:
        if item["value"] not in seen:
            item["rank"] = len(unique_candidates) + 1
            unique_candidates.append(item)
            seen.add(item["value"])

        if len(unique_candidates) >= 10:
            break

    best = unique_candidates[0]

    return (
        best["value"],
        best["ocr_confidence"],
        best["score"],
        unique_candidates
    )


def letterbox(image, new_shape=960):
    h, w = image.shape[:2]

    scale = min(new_shape / h, new_shape / w)

    nh = int(h * scale)
    nw = int(w * scale)

    resized = cv2.resize(image, (nw, nh))

    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)

    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2

    canvas[top:top + nh, left:left + nw] = resized

    return canvas, scale, left, top


def preprocess_detector(image):
    img, scale, pad_x, pad_y = letterbox(image, IMG_SIZE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, scale, pad_x, pad_y


def xywh_to_xyxy(box):
    x, y, w, h = box

    return [
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ]


def postprocess(outputs, original_image, scale, pad_x, pad_y):
    predictions = outputs[0]

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = []
    scores = []
    class_ids = []

    h_img, w_img = original_image.shape[:2]

    for pred in predictions:
        box = pred[:4]
        class_scores = pred[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONF_THRESH:
            continue

        x1, y1, x2, y2 = xywh_to_xyxy(box)

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = max(0, min(w_img, int(x1)))
        y1 = max(0, min(h_img, int(y1)))
        x2 = max(0, min(w_img, int(x2)))
        y2 = max(0, min(h_img, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESH,
        IOU_THRESH
    )

    detections = []

    if len(indices) > 0:
        indices = np.array(indices).flatten()

        for i in indices:
            x, y, w, h = boxes[i]

            detections.append({
                "class_id": class_ids[i],
                "class_name": CLASS_NAMES[class_ids[i]],
                "det_confidence": float(scores[i]),
                "box": [x, y, x + w, y + h]
            })

    return detections


def get_vehicle_type(detections):
    vehicle_priority = ["cars", "motorcyle", "truck"]

    for vehicle in vehicle_priority:
        for det in detections:
            if det["class_name"] == vehicle:
                return vehicle

    return "Unknown"


def crop_with_large_padding(image, box):
    h_img, w_img = image.shape[:2]

    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * 0.25)
    pad_y = int(box_h * 0.45)

    pad_x = max(20, min(pad_x, 70))
    pad_y = max(15, min(pad_y, 50))

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w_img, x2 + pad_x)
    crop_y2 = min(h_img, y2 + pad_y)

    return image[crop_y1:crop_y2, crop_x1:crop_x2]


def process_image(image):
    start_time = time.time()

    input_tensor, scale, pad_x, pad_y = preprocess_detector(image)

    outputs = session.run(
        None,
        {input_name: input_tensor}
    )

    detections = postprocess(
        outputs,
        image,
        scale,
        pad_x,
        pad_y
    )

    vehicle_type = get_vehicle_type(detections)

    best_plate = "Plate Unreadable"
    best_raw_plate = ""
    best_ocr_conf = 0.0
    best_det_conf = 0.0
    best_final_score = 0.0
    best_candidates = []
    best_box = None

    for det in detections:
        if det["class_id"] != LICENSE_PLATE_CLASS_ID:
            continue

        plate_crop = crop_with_large_padding(
            image,
            det["box"]
        )

        plate_text, ocr_conf, final_score, plate_candidates = recognize_plate(plate_crop)

        if plate_text:
            if ocr_conf > best_ocr_conf:
                best_raw_plate = plate_text
                best_plate = format_plate(plate_text)
                best_ocr_conf = ocr_conf
                best_det_conf = det["det_confidence"]
                best_final_score = final_score
                best_candidates = plate_candidates
                best_box = det["box"]

            elif ocr_conf == best_ocr_conf and final_score > best_final_score:
                best_raw_plate = plate_text
                best_plate = format_plate(plate_text)
                best_ocr_conf = ocr_conf
                best_det_conf = det["det_confidence"]
                best_final_score = final_score
                best_candidates = plate_candidates
                best_box = det["box"]

    process_time = time.time() - start_time

    return {
        "plate": best_plate,
        "raw_plate": best_raw_plate,
        "vehicle_type": vehicle_type,
        "det_confidence": round(best_det_conf, 3),
        "ocr_confidence": round(best_ocr_conf, 3),
        "final_score": round(best_final_score, 3),
        "plate_box": {
            "xmin": best_box[0],
            "ymin": best_box[1],
            "xmax": best_box[2],
            "ymax": best_box[3]
        } if best_box else None,
        "plate_candidates": best_candidates,
        "processing_time_seconds": round(process_time, 3)
    }


@app.get("/")
def root():
    return {
        "message": "ALPR API is running",
        "endpoint": "POST /recognize",
        "model": MODEL_PATH,
        "selection_priority": [
            "ocr_confidence",
            "final_score",
            "regex_validity"
        ],
        "response_fields": [
            "plate",
            "raw_plate",
            "vehicle_type",
            "det_confidence",
            "ocr_confidence",
            "final_score",
            "plate_box",
            "plate_candidates",
            "processing_time_seconds"
        ]
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "plate": "Plate Unreadable",
                    "raw_plate": "",
                    "vehicle_type": "Unknown",
                    "det_confidence": 0.0,
                    "ocr_confidence": 0.0,
                    "final_score": 0.0,
                    "plate_box": None,
                    "plate_candidates": [],
                    "processing_time_seconds": 0.0
                }
            )

        result = process_image(image)

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "plate": "Plate Unreadable",
                "raw_plate": "",
                "vehicle_type": "Unknown",
                "det_confidence": 0.0,
                "ocr_confidence": 0.0,
                "final_score": 0.0,
                "plate_box": None,
                "plate_candidates": [],
                "processing_time_seconds": 0.0,
                "error": str(e)
            }
        )