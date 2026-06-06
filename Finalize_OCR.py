import re
import time
from ultralytics import YOLO
from collections import defaultdict, deque
import easyocr
import cv2

start_time = time.time()
 
model = YOLO("models/license_plate_bestM.pt")
reader = easyocr.Reader(["en"], gpu=True)

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


def format_plate(text):
    match = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})$", text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    return text


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
            text = re.sub(r"[^A-Z0-9]", "", text.upper())

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

    _, otsu = cv2.threshold(
        clahe_img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    inverted = cv2.bitwise_not(otsu)

    return [
        gray,
        clahe_img,
        otsu,
        inverted
    ]


def recognize_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    variants = preprocess_variants(plate_crop)

    best_text = ""
    best_score = 0.0

    for variant in variants:
        ocr_results = ocr_with_confidence(variant)

        if not ocr_results:
            continue

        # Case 1: Gabungkan fragment OCR
        joined_text = "".join([text for text, conf in ocr_results])
        joined_text = re.sub(r"[^A-Z0-9]", "", joined_text.upper())

        avg_conf = sum([conf for text, conf in ocr_results]) / len(ocr_results)

        if len(joined_text) >= 4:
            regex_bonus = 1.0 if plate_pattern.match(joined_text) else 0.6
            score = avg_conf * regex_bonus

            if score > best_score:
                best_score = score
                best_text = joined_text

        # Case 2: Cek satu-satu hasil OCR
        for text, conf in ocr_results:
            text = re.sub(r"[^A-Z0-9]", "", text.upper())

            if len(text) < 4:
                continue

            regex_bonus = 1.0 if plate_pattern.match(text) else 0.6
            score = conf * regex_bonus

            if score > best_score:
                best_score = score
                best_text = text

    return best_text


input_foto = "input3.jpg"
output_foto = "output3.jpg"

img = cv2.imread(input_foto)

if img is None:
    raise ValueError("Input gambar tidak ditemukan")

height, width = img.shape[:2]
frame = img.copy()

CONF_THRESH = 0.25
PAD = 12

results = model(frame, verbose=False)

for r in results:
    for box in r.boxes:
        conf = float(box.conf.item())

        if conf < CONF_THRESH:
            continue

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

            print(
                "DETECTED:",
                display_text,
                "| YOLO CONF:",
                round(conf, 2)
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
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
