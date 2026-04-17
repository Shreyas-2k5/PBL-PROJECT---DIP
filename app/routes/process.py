from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import cv2

from app.main import process_image, process_video   # ✅ USE THIS

router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.post("/image")
async def process_image_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(file_path)

    result, _ = process_image(image)

    output_path = os.path.join(OUTPUT_DIR, f"processed_{file.filename}")
    cv2.imwrite(output_path, result)

    return FileResponse(output_path, media_type="image/jpeg")


@router.post("/video")
async def process_video_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = os.path.join(OUTPUT_DIR, f"processed_{file.filename}")

    process_video(file_path, output_path)

    return FileResponse(output_path, media_type="video/mp4")