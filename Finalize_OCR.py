import re
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_PATH = "models/vehicle_plate_yolo11s_best.pt"
INPUT_IMAGE = "input5.jpg"
OUTPUT_IMAGE = "output_trocr.jpg"

LICENSE_PLATE_CLASS_ID = 0
CONF_THRESH = 0.35
PAD = 16

plate_pattern = re.compile(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$")


yolo_model = YOLO(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
).to(device)


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


def preprocess_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return rgb


def recognize_plate_trocr(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    processed = preprocess_plate(plate_crop)

    pil_image = Image.fromarray(processed)

    pixel_values = processor(
        images=pil_image,
        return_tensors="pt"
    ).pixel_values.to(device)

    generated_ids = trocr_model.generate(pixel_values)

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    text = clean_text(text)
    text = fix_plate_by_position(text)

    return text



img = cv2.imread(INPUT_IMAGE)

if img is None:
    raise ValueError("Input gambar tidak ditemukan")

height, width = img.shape[:2]
frame = img.copy()

results = yolo_model(frame, verbose=False)

for r in results:
    for box in r.boxes:
        conf = float(box.conf.item())
        cls = int(box.cls.item())

        if conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if cls != LICENSE_PLATE_CLASS_ID:
            continue

        crop_x1 = max(0, x1 - PAD)
        crop_y1 = max(0, y1 - PAD)
        crop_x2 = min(width, x2 + PAD)
        crop_y2 = min(height, y2 + PAD)

        plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        cv2.imwrite("debug_trocr_crop.jpg", plate_crop)

        plate_text = recognize_plate_trocr(plate_crop)

        if plate_text:
            display_text = format_plate(plate_text)
        else:
            display_text = "Plate Unreadable"

        print("PLATE:", display_text, "| YOLO CONF:", round(conf, 2))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            display_text,
            (x1, y1 - 10 if y1 > 30 else y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

cv2.imwrite(OUTPUT_IMAGE, frame)
print("Saved:", OUTPUT_IMAGE)