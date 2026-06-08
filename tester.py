import re
import cv2
import time
import numpy as np
import onnxruntime as ort
from paddleocr import PaddleOCR


MODEL_PATH = "models/vehicle_plate_yolov9tiny_best.onnx"
IMAGE_PATH = "input3.jpg"
OUTPUT_PATH = "output3.jpg"

IMG_SIZE = 640
CONF_THRESH = 0.35
IOU_THRESH = 0.45
PAD = 12

CLASS_NAMES = [
    "License_Plate",
    "cars",
    "motorcyle",
    "truck"
]

LICENSE_PLATE_CLASS_ID = 0

plate_pattern = re.compile(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$")


ocr = PaddleOCR(
     use_angle_cls=True,
    lang="en",
    show_log=False
)


def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def format_plate(text):
    match = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})$", text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    return text


def fix_plate_by_position(text):
    text = clean_text(text)

    to_digit = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1",
        "Z": "2", "S": "5",
        "G": "6", "B": "8"
    }

    to_letter = {
        "0": "O", "1": "I",
        "2": "Z", "5": "S",
        "6": "G", "8": "B"
    }

    chars = list(text)

    for i in range(len(chars)):
        if 1 <= i <= 5:
            chars[i] = to_digit.get(chars[i], chars[i])

    for i in range(len(chars)):
        if i >= 5:
            chars[i] = to_letter.get(chars[i], chars[i])

    return "".join(chars)


def preprocess_plate_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced

def extract_indonesian_plate(text):
    text = clean_text(text)

    pattern = re.search(
        r"[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}",
        text
    )

    if pattern:
        return pattern.group(0)

    return text


def recognize_plate(crop):
    if crop is None or crop.size == 0:
        return "", 0.0

    processed = preprocess_plate_for_ocr(crop)

    result = ocr.ocr(processed,cls=True)

    if not result or not result[0]:
        return "", 0.0

    texts = []
    confs = []

    for line in result[0]:
        text = line[1][0]
        conf = float(line[1][1])

        text = clean_text(text)

        if text:
            texts.append(text)
            confs.append(conf)

    if not texts:
        return "", 0.0

    joined_text = clean_text("".join(texts))
    fixed_text = fix_plate_by_position(joined_text)
    fixed_text = extract_indonesian_plate(fixed_text)

    avg_conf = sum(confs) / len(confs)

    return fixed_text, avg_conf


def letterbox(image, new_shape=640):
    h, w = image.shape[:2]

    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(h * scale), int(w * scale)

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
        for i in indices:
            x, y, w, h = boxes[i]

            detections.append({
                "class_id": class_ids[i],
                "class_name": CLASS_NAMES[class_ids[i]],
                "det_confidence": scores[i],
                "box": [x, y, x + w, y + h]
            })

    return detections


def draw_label(frame, x1, y1, x2, y2, text, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        text,
        (x1, y1 - 10 if y1 > 30 else y2 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


def main():
    start_time = time.time()

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise ValueError(f"Gambar tidak ditemukan: {IMAGE_PATH}")

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

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

    frame = image.copy()
    results = []

    h_img, w_img = image.shape[:2]

    for idx, det in enumerate(detections):
        class_id = det["class_id"]
        class_name = det["class_name"]
        det_conf = det["det_confidence"]
        x1, y1, x2, y2 = det["box"]

        if class_id != LICENSE_PLATE_CLASS_ID:
            label = f"{class_name} {det_conf:.2f}"
            draw_label(frame, x1, y1, x2, y2, label, (255, 0, 0))
            continue

        crop_x1 = max(0, x1 - PAD)
        crop_y1 = max(0, y1 - PAD)
        crop_x2 = min(w_img, x2 + PAD)
        crop_y2 = min(h_img, y2 + PAD)

        plate_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        crop_path = f"plate_crop_{idx}.jpg"
        cv2.imwrite(crop_path, plate_crop)

        plate_text, ocr_conf = recognize_plate(plate_crop)

        if plate_text:
            display_text = format_plate(plate_text)
        else:
            display_text = "Plate Unreadable"

        label = f"{display_text} | D:{det_conf:.2f} OCR:{ocr_conf:.2f}"

        draw_label(
            frame,
            x1,
            y1,
            x2,
            y2,
            label,
            (0, 255, 0)
        )

        results.append({
            "plate": display_text,
            "raw_plate": plate_text,
            "det_confidence": round(det_conf, 4),
            "ocr_confidence": round(ocr_conf, 4),
            "box": [x1, y1, x2, y2],
            "crop_file": crop_path
        })

        print("PLATE:", display_text)
        print("DET CONF:", round(det_conf, 4))
        print("OCR CONF:", round(ocr_conf, 4))
        print("CROP:", crop_path)
        print("-" * 40)

    cv2.imwrite(OUTPUT_PATH, frame)

    end_time = time.time()
    process_time = end_time - start_time

    response = {
        "success": True,
        "processing_time_seconds": round(process_time, 3),
        "total_detected": len(results),
        "results": results,
        "output_file": OUTPUT_PATH
    }

    print(response)


if __name__ == "__main__":
    main()