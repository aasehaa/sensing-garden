from picamera2 import Picamera2, controls
from ultralytics import YOLO
import time
from datetime import datetime
import cv2
import numpy as np
import os


# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.set_controls({"AfMode": 0})

# if you want to configure the focus
#picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})



picam2.configure(camera_config)
picam2.start()

# Load the YOLO model
model = YOLO("./models/float32_workshop2.tflite")

# define output folder
output_folder = "detections"


# Function to capture image, run YOLO, and save results


def capture_and_process():
    # Capture image
    image = picam2.capture_array()

    # Run YOLO model
    results = model(image)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # adjust color format from Blue-Green-Red (BGR) to Red-Green-Blue (RGB)(Raspberry Pi camera)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # Save image
    cv2.imwrite(os.path.join(output_folder, f"image_{timestamp}.jpg"), image)

    # Save detections
    with open(os.path.join(output_folder, f"detections_{timestamp}.txt"), "w") as f:
        for r in results:
            for box in r.boxes:
                f.write(f"{r.names[int(box.cls)]} {box.conf.item():.2f} {box.xyxy[0].tolist()}\n")

# Main loop
while True:
    capture_and_process()
    time.sleep(10)
