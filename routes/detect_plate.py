from fastapi import APIRouter, UploadFile, File
from Services.yolo_services import YOLOService
import tempfile
import shutil
import cv2
import time

router = APIRouter()

yolo_service = YOLOService()


class DetectPlateController:

    @staticmethod
    @router.post("/detect-plate")
    async def detect_plate(
        file: UploadFile = File(...)
    ):

        start_time = time.time()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file
            )

            temp_path = temp_file.name

        frame = cv2.imread(temp_path)

        results = yolo_service.detect_plate(
            frame
        )

        end_time = time.time()

        return {
            "success": True,
            "processing_time_seconds": round(
                end_time - start_time,
                2
            ),
            "total_detected": len(results),
            "results": results
        }