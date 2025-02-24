import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet152, resnet50
import torch.nn as nn
from picamera2 import Picamera2
import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} executed in {elapsed_time:.6f} seconds")
        return result
    return wrapper

def set_up():
    global device, yolo_model, classification_model, classification_transform, state_dict, num_classes, picam2 # declare the variables as global

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set the device
    print(f"Using device: {device}")
    
    #yolo_model = YOLO("insect-yolov8.pt") # load the yolo model
    yolo_model = YOLO("generic-insect-v2.pt")
    classification_model = resnet50(pretrained=False) # load the classification model
    #num_classes = 22 # set the number of classes
    num_classes = 9 # 9 species test

    classification_model.fc = nn.Linear(classification_model.fc.in_features, num_classes) # set the number of output classes  
    #state_dict = torch.load("species-resnet152-v1.pth", map_location=device) # load the state dictionary
    state_dict = torch.load("insect_resnet152.pth", map_location=device)
    classification_model.load_state_dict(state_dict) # load the state dictionary
    classification_model.to(device) # move the model to the device
    classification_model.eval() # set the model to evaluation mode
    classification_transform = transforms.Compose([
        transforms.Resize(640), # resize images before classification will improve inferrence but loose resolution
        transforms.ToTensor(), # convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # normalize the image
                             std=[0.229, 0.224, 0.225]) # normalize the image
    ])

    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"format": 'RGB888', "size": (2592, 2592)}) # still images with higher resolutions
    #camera_config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1080, 1080)}) # video with lower resolution (highest possible)
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    picam2.configure(camera_config)
    picam2.start()


@timer_decorator
def frame_processing(species_names, classification_counts):

    frame = picam2.capture_array() # capture the frame

    results = yolo_model.track(source=frame, conf=0.3, iou=0.5, tracker='botsort.yaml', persist=True) # track the frame

    frame = results[0].orig_img # get the frame
    detections = results[0].boxes # get the detections         
    for box in detections:

        try: # try to get the box coordinates
            xyxy = box.xyxy.cpu().numpy().flatten().astype(int)
            x1, y1, x2, y2 = xyxy[:4]
        except Exception as e: # if there is an error, print the error and continue
            print(f"Error processing box coordinates: {e}")
            continue

        h, w = frame.shape[:2] # get the height and width of the frame

        track_id = getattr(box, 'id', None) # get the track id
        if track_id is not None: # if the track id is not None, try to convert the track id to an integer
            try:
                track_id = int(track_id.item() if hasattr(track_id, 'item') else track_id)
            except Exception as e: # if there is an error, print the error and continue
                track_id = "N/A"
        else: # if the track id is None, set the track id to "N/A"
            track_id = "N/A"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # draw the rectangle on the frame
        insect_crop = frame[y1:y2, x1:x2] # get the insect crop
        insect_crop_rgb = cv2.cvtColor(insect_crop, cv2.COLOR_BGR2RGB) # convert the insect crop to rgb
        pil_img = Image.fromarray(insect_crop_rgb) # convert the insect crop to a PIL image
        input_tensor = classification_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = classification_model(input_tensor) # get the outputs from the classification model


        predicted_class_idx = outputs.argmax(dim=1).item() # get the predicted class index
        species_name = species_names[predicted_class_idx] if predicted_class_idx < len(species_names) else "Unknown" # get the species name
        if track_id not in classification_counts: # if the track id is not in the classification counts, add it to the classification counts
            classification_counts[track_id] = {}
        if species_name not in classification_counts[track_id]: # if the species name is not in the classification counts, add it to the classification counts
            classification_counts[track_id][species_name] = 0
        classification_counts[track_id][species_name] += 1

        label = f"ID: {track_id} | {species_name}" # get the label
        print( label)
        font = cv2.FONT_HERSHEY_SIMPLEX # get the font
        font_scale = 0.7 # get the font scale
        thickness = 2 # get the thickness
        org = (x1, y1 - 10) # get the org
        cv2.putText(frame, label, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, label, org, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    #cv2.imshow("Insect Tracking & Classification", frame) # show the frame

    return classification_counts

if __name__ == "__main__":

    set_up() # set up the variables

    species_names = [ #species that resnet is trained on 
        "Aglais io",
        "Aglais urticae",
        "Bombus hortorum",
        "Bombus lapidarius",
        "Bombus lucorum",
        "Bombus monticola",
        "Bombus pascuorum",
        "Colias croceus",
        "Colletes hederae",
        "Episyrphus balteatus",
        "Eristalis tenax",
        "Gonepteryx rhamni",
        "Myathropa florea",
        "Pieris brassicae",
        "Pieris rapae",
        "Polygonia c-album",
        "Rhingia campestris",
        "Syrphus ribesii",
        "Vanessa atalanta",
        "Vanessa cardui",
        "Vespa crabro",
        "Vespula vulgaris"
    ]

    species_names_9species=[
        "Coccinella septempunctata",
        "Apis mellifera",
        "Bombus lapidarius",
        "Bombus terrestris",
        "Eupeodes corollae",
        "Episyrphus balteatus",
        "Aglais urticae",
        "Vespula vulgaris",
        "Eristalis tenax"
    ]


    while True: 

        classification_counts = {}

        classification_counts = frame_processing(species_names_9species, classification_counts) # process the frame

        if cv2.waitKey(1) & 0xFF == ord('q'): # if the q key is pressed, break the loop
            break

    
    print(classification_counts)

    cv2.destroyAllWindows() # destroy all windows
    picam2.stop() # stop the camera

    
