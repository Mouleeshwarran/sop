from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np

from backend.detector import detect


app = FastAPI(title="R.A.H.A.T Backend")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "R.A.H.A.T backend is running"
    }


# =========================================================
# DETECTION API
# =========================================================

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    image_bytes = await file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    # -----------------------------------------------------
    # Validate image
    # -----------------------------------------------------

    if frame is None:

        return {
            "success": False,
            "error": "Invalid image"
        }


    # -----------------------------------------------------
    # Run complete detection + navigation system
    # -----------------------------------------------------

    result = detect(frame)


    # -----------------------------------------------------
    # Return complete result to frontend
    # -----------------------------------------------------

    return {
        "success": True,

        "detections": result["detections"],

        "navigation": result["navigation"],

        "frame": result["frame"],

        "object_count": result["object_count"]
    }