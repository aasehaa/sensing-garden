from ultralytics import YOLO
import os
import time
import cv2
from save_cropped_image import save_hq_detection
from utils import log_detection_information
from PIL import Image


'''
using OpenCV for image processing and writing
'''


# load the yolo model
model = YOLO("./models/genericInsect/best.pt")

processed_images = set()
print("start")

while True:
    for filename in os.listdir("image_detected_frames"):
        if filename.endswith(".jpg") and filename not in processed_images:
            image_path = os.path.join("image_detected_frames", filename)

            image = cv2.imread(image_path)
            print('before')
            results = model(image)
            print('after')
            
            # make sure folder for detections exist
            #os.makedirs("test_image_detected_frames", exist_ok=True)
            #detection_path = os.path.join("test_image_detected_frames", filename.replace(".jpg", ".txt"))
            print('test')

            if len(results[0].boxes) > 0: # checks if there are any bounding boxes
                print('Object detected! YAY')
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"testimage_detected_frames/{timestamp}.jpg"

                # save detected frame
                cv2.imwrite(filename, image) 

                # log detection information
                #log_detection_information("testimg_detections.log", results, timestamp, "testimage")
                
                for detection in results[0].boxes.data:
                
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                    left = int(x_min)
                    upper = int(y_min)
                    right = int(x_max)
                    lower = int(y_max)

                    print("left, upper, right, lower: ", left, upper, right, lower)
                    original_image = Image.open(image_path)
                    # Crop the image
                    cropped_image = original_image.crop((left, upper, right, lower))

                    # Save the cropped image
                    cropped_image.save(f"testimage_hq_detections/{timestamp}.jpg")

            else:
                print("No objects detected in this frame.")
            
            print('image processed')
            processed_images.add(filename)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    
    #time.sleep(2) # change interval for checking new detections

