from ultralytics import YOLO
import os
import time
import cv2

'''
using OpenCV for image processing and writing
'''

def process_new_images():
    # load the yolo model
    model = YOLO("./models/genericInsect/best.pt")

    processed_images = set()

    while True:
        for filename in os.listdir("frames"):
            if filename.endswith(".jpg") and filename not in processed_images:
                image_path = os.path.join("frames", filename)
                image = cv2.imread(image_path)
                
                results = model(image)
                
                # make sure folder for detections exist
                os.makedirs("detections", exist_ok=True)
                detection_path = os.path.join("detections", filename.replace(".jpg", ".txt"))
                
                with open(detection_path, "w") as f:
                    for r in results:
                        for box in r.boxes:
                            f.write(f"{r.names[int(box.cls)]} {box.conf.item():.2f} {box.xyxy[0].tolist()}\n")
                
                processed_images.add(filename)
        
        time.sleep(10)  # Check for new images every 5 seconds

