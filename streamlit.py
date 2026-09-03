import re
import cv2
import time
import numpy as np
import onnxruntime as ort
import streamlit as st

from PIL import Image
from paddleocr import PaddleOCR



MODEL_PATH = "models/vehicle_plate_yolov9tiny_best.onnx"

IMG_SIZE = 640
CONF_THRESH = 0.35
IOU_THRESH = 0.45
PAD = 4

CLASS_NAMES = [
    "License_Plate",
    "cars",
    "motorcyle",
    "truck"
]

LICENSE_PLATE_CLASS_ID = 0

plate_pattern = re.compile(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$")
plate_search_pattern = re.compile(r"[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}")


@st.cache_resource
def load_models():
    ocr_model = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False
    )

    providers = ort.get_available_providers()

    if "CUDAExecutionProvider" in providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=providers
    )

    input_name = session.get_inputs()[0].name

    return session, input_name, ocr_model, providers


session, input_name, ocr, active_providers = load_models()


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


def fix_plate_by_position(text):
    text = clean_text(text)

    if len(text) < 4:
        return text

    if plate_pattern.match(text):
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
    match = plate_search_pattern.search(text)

    if match:
        return match.group(0)

    return text


def score_plate(text, conf):
    text = clean_text(text)

    if plate_pattern.fullmatch(text):
        return conf + 0.50

    if plate_search_pattern.search(text):
        return conf + 0.25

    if len(text) >= 4:
        return conf

    return conf - 0.50


def preprocess_variants_for_ocr(crop):
    variants = []

    if crop is None or crop.size == 0:
        return variants

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    clahe_img = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    ).apply(resized)

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    sharp = cv2.filter2D(clahe_img, -1, kernel)

    _, otsu = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    variants.append(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

    return variants


def run_ocr(image):
    try:
        result = ocr.ocr(image, cls=True)

        if not result or not result[0]:
            return "", 0.0

        texts = []
        confs = []

        for line in result[0]:
            text = clean_text(line[1][0])
            conf = float(line[1][1])

            if text:
                texts.append(text)
                confs.append(conf)

        if not texts:
            return "", 0.0

        joined_text = clean_text("".join(texts))
        avg_conf = sum(confs) / len(confs)

        return joined_text, avg_conf

    except Exception:
        return "", 0.0


def recognize_plate(crop):
    variants = preprocess_variants_for_ocr(crop)

    best_text = ""
    best_conf = 0.0
    best_score = -999

    for variant in variants:
        raw_text, conf = run_ocr(variant)

        if not raw_text:
            continue

        fixed_text = fix_plate_by_position(raw_text)
        extracted_text = extract_indonesian_plate(fixed_text)

        current_score = score_plate(extracted_text, conf)

        if current_score > best_score:
            best_score = current_score
            best_text = extracted_text
            best_conf = conf

    return best_text, best_conf


def letterbox(image, new_shape=640):
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
                "det_confidence": scores[i],
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


def draw_label(frame, x1, y1, x2, y2, text, color):
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        text,
        (x1, y1 - 10 if y1 > 30 else y2 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


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

    frame = image.copy()
    h_img, w_img = image.shape[:2]

    best_plate = "Plate Unreadable"
    best_score = -999

    for det in detections:
        class_id = det["class_id"]

        x1, y1, x2, y2 = det["box"]
        det_conf = det["det_confidence"]

        if class_id != LICENSE_PLATE_CLASS_ID:
            draw_label(
                frame,
                x1,
                y1,
                x2,
                y2,
                det["class_name"],
                (255, 0, 0)
            )
            continue

        crop_x1 = max(0, x1 - PAD)
        crop_y1 = max(0, y1 - PAD)
        crop_x2 = min(w_img, x2 + PAD)
        crop_y2 = min(h_img, y2 + PAD)

        plate_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        plate_text, ocr_conf = recognize_plate(plate_crop)

        if plate_text:
            display_text = format_plate(plate_text)
            current_score = det_conf + ocr_conf

            if current_score > best_score:
                best_score = current_score
                best_plate = display_text

            draw_label(
                frame,
                x1,
                y1,
                x2,
                y2,
                display_text,
                (0, 255, 0)
            )
        else:
            draw_label(
                frame,
                x1,
                y1,
                x2,
                y2,
                "Plate Unreadable",
                (0, 255, 255)
            )

    process_time = time.time() - start_time

    return {
        "plate": best_plate,
        "vehicle_type": vehicle_type,
        "processing_time_seconds": round(process_time, 3),
        "output_image": frame
    }


st.set_page_config(
    page_title="ALPR Vehicle Recognition",
    page_icon="🚗",
    layout="wide"
)

st.title("ALPR Vehicle Recognition")
st.caption("Snapshot ALPR dengan jarak efektif ±1 meter")

with st.sidebar:
    st.header("System Info")
    st.write("Model: YOLOv9-tiny ONNX")
    st.write("OCR: PaddleOCR")
    st.write("Effective Distance: ±1 meter")
    st.write("ONNX Provider:")
    st.code(str(active_providers))

uploaded_file = st.file_uploader(
    "Upload gambar kendaraan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    st.subheader("Input Image")
    st.image(
        image_rgb,
        use_container_width=True
    )

    if st.button("Recognize Plate"):
        with st.spinner("Processing ALPR..."):
            result = process_image(image_bgr)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Plate",
                result["plate"]
            )

        with col2:
            st.metric(
                "Vehicle Type",
                result["vehicle_type"]
            )

        with col3:
            st.metric(
                "Processing Time",
                f'{result["processing_time_seconds"]} sec'
            )

        output_rgb = cv2.cvtColor(
            result["output_image"],
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Output Image")
        st.image(
            output_rgb,
            use_container_width=True
        )
else:
    st.info("Upload gambar kendaraan terlebih dahulu.")