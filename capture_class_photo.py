import cv2
import os
from datetime import datetime

def capture_photo(subject_id):
    cam = cv2.VideoCapture(0)

    ret, frame = cam.read()
    cam.release()

    if not ret:
        raise Exception("Camera failed")

    os.makedirs("uploads", exist_ok=True)

    filename = f"uploads/class_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)

    return filename
