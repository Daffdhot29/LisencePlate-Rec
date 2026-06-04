import re
import time
from ultralytics import YOLO
from collections import defaultdict, deque
import easyocr
import cv2
import numpy as np

start_time = time.time()




model = YOLO("models/vehicle_plate_yolo11s_best.pt")
reader = easyocr.Reader(["en"], gpu=True)


LICENSE_PLATE_CLASS_ID = 0

CLASS_NAMES = {
    0: "License Plate",
    1: "Car",
    2: "Motorcycle",
    3: "Truck"
}

plate_pattern = re.compile(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$")

plate_history = defaultdict(lambda: deque(maxlen=15))
plate_final = {}


def get_box(x1, x2, y1, y2):
    return f"{int(x1 // 15)}_{int(x2 // 15)}_{int(y1 // 15)}_{int(y2 // 15)}"


def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)

        most_common = max(
            set(plate_history[box_id]),
            key=plate_history[box_id].count
        )

        plate_final[box_id] = most_common

    return plate_final.get(box_id, "")


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

    fixed = "".join(chars)
    chars = list(fixed)

    for i in range(len(chars)):
        if i >= 5:
            chars[i] = to_letter.get(chars[i], chars[i])

    return "".join(chars)



def ocr_with_confidence(image):
    try:
        result = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        output = []

        for bbox, text, conf in result:
            text = clean_text(text)

            if text:
                output.append((text, conf))

        return output

    except Exception as e:
        print("OCR ERROR:", e)
        return []


def preprocess_variants(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(blur)

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

    inverted_otsu = cv2.bitwise_not(otsu)
    inverted_adaptive = cv2.bitwise_not(adaptive)

    return [
        gray,
        clahe_img,
        sharp,
        otsu,
        adaptive,
        inverted_otsu,
        inverted_adaptive
    ]


def recognize_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    variants = preprocess_variants(plate_crop)

    best_text = ""
    best_score = 0.0
    raw_candidate = ""

    for variant in variants:
        ocr_results = ocr_with_confidence(variant)

        if not ocr_results:
            continue

        joined_text = "".join([text for text, conf in ocr_results])
        joined_text = clean_text(joined_text)

        if len(joined_text) >= 4 and not raw_candidate:
            raw_candidate = joined_text

        fixed_joined = fix_plate_by_position(joined_text)

        avg_conf = sum([conf for text, conf in ocr_results]) / len(ocr_results)

        if plate_pattern.match(fixed_joined):
            score = avg_conf + 0.25

            if score > best_score:
                best_score = score
                best_text = fixed_joined

        for text, conf in ocr_results:
            raw_text = clean_text(text)

            if len(raw_text) >= 4 and not raw_candidate:
                raw_candidate = raw_text

            fixed_text = fix_plate_by_position(raw_text)

            if plate_pattern.match(fixed_text):
                score = conf + 0.15

                if score > best_score:
                    best_score = score
                    best_text = fixed_text

    if best_text:
        return best_text

   
    return raw_candidate



input_foto = "input5.jpg"
output_foto = "output5.jpg"

img = cv2.imread(input_foto)

if img is None:
    raise ValueError("Input gambar tidak ditemukan")

height, width = img.shape[:2]
frame = img.copy()

CONF_THRESH = 0.35
PAD = 16


results = model(frame, verbose=False)

for r in results:
    for box in r.boxes:
        conf = float(box.conf.item())
        cls = int(box.cls.item())

        if conf < CONF_THRESH:
            continue

        class_name = CLASS_NAMES.get(cls, "Unknown")

        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

      
        if cls != LICENSE_PLATE_CLASS_ID:
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10 if y1 > 30 else y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            print("VEHICLE:", class_name, "| CONF:", round(conf, 2))
            continue



 
        crop_x1 = max(0, x1 - PAD)
        crop_y1 = max(0, y1 - PAD)
        crop_x2 = min(width, x2 + PAD)
        crop_y2 = min(height, y2 + PAD)

        plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        cv2.imwrite("debug_plate_crop.jpg", plate_crop)

        text = recognize_plate(plate_crop)

        stable_id = get_box(x1, x2, y1, y2)
        final_text = get_stable_plate(stable_id, text)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        if final_text:
            display_text = format_plate(final_text)
        else:
            display_text = "Plate Unreadable"

        print(
            "PLATE:",
            display_text,
            "| YOLO CONF:",
            round(conf, 2)
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(
            display_text,
            font,
            font_scale,
            thickness
        )

        text_x = x1
        text_y = y1 - 10

        if text_y - th - baseline < 0:
            text_y = y2 + th + 10

        cv2.rectangle(
            frame,
            (text_x, text_y - th - baseline),
            (text_x + tw + 8, text_y + baseline),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            display_text,
            (text_x + 4, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )


cv2.imwrite(output_foto, frame)

end_time = time.time()

print(f"\nPROCESS TIME: {end_time - start_time:.2f} sec")
print("Saved:", output_foto)