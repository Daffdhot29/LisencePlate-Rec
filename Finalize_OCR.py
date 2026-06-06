import re
import cv2
import time
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_PATH = "models/anpr_vehicle_best.pt"
INPUT_IMAGE = "input4.jpg"
OUTPUT_IMAGE = "output4.jpg"

LICENSE_PLATE_CLASS_ID = 0
CONF_THRESH = 0.35
PAD = 16

plate_pattern = re.compile(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$")



start_load_time = time.time()

yolo_model = YOLO(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

trocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
).to(device)

trocr_model.eval()

load_time = time.time() - start_load_time
print(f"MODEL LOAD TIME: {load_time:.3f}s")



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
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }

    to_letter = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
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
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return rgb



def recognize_plate_trocr(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return "", 0.0

    ocr_start = time.time()

    processed = preprocess_plate(plate_crop)
    pil_image = Image.fromarray(processed)

    pixel_values = processor(
        images=pil_image,
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = trocr_model.generate(
            pixel_values,
            max_length=12,
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    text = clean_text(text)
    text = fix_plate_by_position(text)

    ocr_time = time.time() - ocr_start

    return text, ocr_time



start_time = time.time()

img = cv2.imread(INPUT_IMAGE)

if img is None:
    raise ValueError(f"Input gambar tidak ditemukan: {INPUT_IMAGE}")

height, width = img.shape[:2]
frame = img.copy()


yolo_start = time.time()
results = yolo_model(frame, verbose=False)
yolo_time = time.time() - yolo_start

total_detected = 0
total_plate_read = 0
total_ocr_time = 0.0

for r in results:
    for box in r.boxes:
        conf = float(box.conf.item())
        cls = int(box.cls.item())

        if conf < CONF_THRESH:
            continue

        if cls != LICENSE_PLATE_CLASS_ID:
            continue

        total_detected += 1

        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        crop_x1 = max(0, x1 - PAD)
        crop_y1 = max(0, y1 - PAD)
        crop_x2 = min(width, x2 + PAD)
        crop_y2 = min(height, y2 + PAD)

        plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        debug_crop_name = f"debug_trocr_crop_{total_detected}.jpg"
        cv2.imwrite(debug_crop_name, plate_crop)

        plate_text, ocr_time = recognize_plate_trocr(plate_crop)

        total_ocr_time += ocr_time

        if plate_text:
            total_plate_read += 1
            display_text = format_plate(plate_text)
        else:
            display_text = "Plate Unreadable"

        print(
            f"PLATE {total_detected}: {display_text} | "
            f"YOLO CONF: {conf:.2f} | "
            f"OCR TIME: {ocr_time:.3f}s"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            display_text,
            (x1, y1 - 10 if y1 > 30 else y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )


cv2.imwrite(OUTPUT_IMAGE, frame)

total_time = time.time() - start_time

print("\n========== SUMMARY ==========")
print(f"Input Image        : {INPUT_IMAGE}")
print(f"Output Image       : {OUTPUT_IMAGE}")
print(f"Total Detected     : {total_detected}")
print(f"Total Plate Read   : {total_plate_read}")
print(f"YOLO Time          : {yolo_time:.3f}s")
print(f"Total OCR Time     : {total_ocr_time:.3f}s")
print(f"Total Process Time : {total_time:.3f}s")
print(f"Model Load Time    : {load_time:.3f}s")
print("=============================")