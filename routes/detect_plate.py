from fastapi import APIRouter, UploadFile, File, HTTPException
from Services.yolo_services import YOLOService
import numpy as np
import cv2
import time
import uuid
import os
import asyncio

router = APIRouter(prefix="/recognize")

yolo_service = YOLOService()

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


def run_detection(frame):
    return yolo_service.detect_plate(frame)


@router.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):

    start_time = time.time()

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        np_img = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")


        debug_path = os.path.join(
            DEBUG_DIR,
            f"{uuid.uuid4().hex}.jpg"
        )

        cv2.imwrite(debug_path, frame)


        detected_plates = await asyncio.to_thread(
            run_detection,
            frame
        )

        end_time = time.time()

        return {
            "success": True,
            "processing_time_seconds": round(end_time - start_time, 3),
            "total_detected": len(detected_plates),
            "results": detected_plates
        }

    except HTTPException as he:
        return {
            "success": False,
            "error": he.detail
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }