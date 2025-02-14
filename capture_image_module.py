from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
from utils import save_hq_detection

# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"format": 'RGB888', "size": (2592, 2592)})
picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
picam2.configure(camera_config)
picam2.start()

# Load the YOLO model
model = YOLO("./models/genericInsect/best.pt")


def capture_and_store():
    
    # capture image
    image = picam2.capture_array()

    # Run detection
    detections = model(image)

    if len(detections[0].boxes) > 0: # checks if there are any bounding boxes
        print('Object detected!')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"image_detected_frames/{timestamp}.jpg"

        # save detected frame
        cv2.imwrite(filename, image) 

        # Log and store detection information
        save_hq_detection("img_detections.log", detections, timestamp, "image")
    else:
        print("No objects detected in this frame.")
    
    #picam2.capture_file(filename)


while True:
    capture_and_store()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
