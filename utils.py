from PIL import Image

def save_hq_detection(logfile: str, detections: list, timestamp: str, input: str):
    timestamp: str
    x_min: float
    y_min: float
    x_max: float
    y_max:float
    input: str

    with open(logfile, "a") as log_file:
            count = 0
            for detection in detections[0].boxes.data:
                x_min, y_min, x_max, y_max, confidence, class_id = detection
 
                class_id = int(class_id)
                confidence = float(confidence)
                log_file.write(f"{timestamp},{class_id},{confidence}, {x_min}, {y_min}, {x_max}, {y_max}\n")

                # Open the original image
                original_image = Image.open(f"{input}_detected_frames/{timestamp}.jpg")

                left = int(x_min)
                upper = int(y_min)
                right = int(x_max)
                lower = int(y_max)

                print("left, upper, right, lower: ", left, upper, right, lower)

                # Crop the image
                cropped_image = original_image.crop((left, upper, right, lower))

                # Save the cropped image
                cropped_image.save(f"{input}_hq_detections/{timestamp}_{count}.jpg")
                
                # count if multiple detections
                count = count + 1