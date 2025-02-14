import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
from utils import save_hq_detection

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1080, 1080)})
picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
picam2.configure(camera_config)
picam2.start()

# Load the YOLO model
model = YOLO("./models/genericInsect/best.pt")

def capture_and_store():
    # Capture frame
    frame = picam2.capture_array()

    # Run detection
    detections = model(frame)

    # Check if there are any detections
    if len(detections[0].boxes) > 0:  # This checks if there are any bounding boxes
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"video_detected_frames/{timestamp}.jpg"

        # save detected frame
        cv2.imwrite(filename, frame) 

        # Log and store detection information
        save_hq_detection("video_detections.log", detections, timestamp, "video")

    else:
        print("No objects detected in this frame.")



while True: 
    capture_and_store()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
