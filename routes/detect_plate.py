from fastapi import APIRouter, UploadFile, File
from Services.yolo_services import YOLOService
import numpy as np
import cv2
import time

router = APIRouter()

yolo_service = YOLOService()


@router.post("/plate")
async def detect_plate(file: UploadFile = File(...)):

    start_time = time.time()


    image_bytes = await file.read()

    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "success": False,
            "message": "Invalid image"
        }

  
    cv2.imwrite("debug_input.jpg", frame)

  
    detected_plates = yolo_service.detect_plate(frame)


    end_time = time.time()

    return {
        "success": True,
        "processing_time_seconds": round(end_time - start_time, 3),
        "total_detected": len(detected_plates),
        "results": detected_plates
    }