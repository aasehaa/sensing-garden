# Guide for setting up the Sensing Garden Monitoring System


Hardware you need to setup this system: 
- RaspberryPi (v.3-5)
- Converter cable for raspberryPi5 if needed
- Camera Module 3 (regular or wide)
- RaspberryPi 27W USB-C Power Supply
- MicroHDMI to HDMI
- Mouse
- Keyboard


## Getting started with RaspberryPi 

Follow the guide to setup your raspberry pi on their official webpage: 

[Official guide on the RaspberryPi Website](https://www.raspberrypi.com/documentation/computers/getting-started.html)


TIPS: Enable [RaspberryPi Connect](https://www.raspberrypi.com/documentation/computers/getting-started.html#raspberry-pi-connect) to be able to connect to your raspberryPi remotely. 


## Camera Module

*Official documentation for the camera module: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf#page=6.10*

---

As of mid-September 2022, Picamera2 is pre-installed in all Raspberry Pi OS images. You can update it with a full system update, or via the terminal with:

```
sudo apt install -y python3-picamera2
``` 

You can check that libcamera is working by opening a command window and typing:

```python
rpicam-hello
```

You should see a camera preview window for about five seconds. If you do not, please refer to the Raspberry Pi camera documentation.


*Note to self: If you're using picamera2 installed through `apt`, you don't need a virtual environment. Just run your script directly with the system's Python*


### First example

The following script will:
1. Open the camera system
2. Generate a camera configuration suitable for preview
3. Configure the camera system with that preview configuration
4. Start the preview window
5. Start the camera running
6. Wait for two seconds and capture a JPEG file (still in the preview resolution)


GUI users should enter:

```python
from picamera2 import Picamera2, Preview
import time
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(2)
picam2.capture_file("test.jpg")

```


## Ultralytics

*Official guide from Ultralytics platform: https://docs.ultralytics.com/guides/raspberry-pi/#install-ultralytics-package*


### Run TensorFlow Lite (tflite) on Raspberry Pi

Running TensorFlow Lite (TFLite) models on a Raspberry Pi offers several key advantages:

- **Improved Performance**: TFLite models are optimized for on-device inference, resulting in faster execution times on resource-constrained devices like the Raspberry Pi14.
- **Reduced Latency**: By processing data locally, TFLite minimizes the need for cloud-based computation, leading to quicker response times in real-time applications13.
- **Offline Capability**: TFLite models can run without an internet connection, making them suitable for remote or disconnected environments1.
- **Power Efficiency**: TFLite's optimization techniques result in lower power consumption, which is crucial for battery-powered or energy-constrained Raspberry Pi projects3.
- **Versatility**: TFLite supports various applications on Raspberry Pi, including computer vision tasks, object detection, and image classification4.
- **Hardware Acceleration**: TFLite can leverage hardware acceleration on the Raspberry Pi, further improving performance for neural network computations34.


First step is to convert your yolo model to tflite: 

1. **Convert model:** 

```python
from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt") # make sure the path is correct for your model

# Export the model to NCNN format
model.export(format="tflite")  # creates 'best_saved_model' folder

```

If you want to check the model sizes and compare after exporting to tflite, you can run this code: 

```python

yolo11n_file = "runs/detect/train/weights/best.pt"
tflite_file = "runs/detect/train/weights/best_saved_model/best_int8.tflite"

def get_model_size(filepath):

  # Get the file size in bytes
  file_size = os.path.getsize(filepath)

  # Convert to MB for readability
  file_size_MB = file_size / (1024 * 1024)

  print(f"Model size: {file_size_MB:.2f} MB")

get_model_size(yolo11n_file)
get_model_size(tflite_file)

```



2. **Use Raspberrypi camera with the model**

There are 2 methods of using the Raspberry Pi Camera to inference YOLO11 models.

**Method 1:**
We can use `picamera2` which comes pre-installed with Raspberry Pi OS to access the camera and inference YOLO11 models.

```python
import cv2
from picamera2 import Picamera2

from ultralytics import YOLO

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO11 model
model = YOLO("models/best_int8.tflite") # edit based on model name above

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()

    # Run ftlite-model inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()

```

**Method 2:**
We need to initiate a TCP stream with `rpicam-vid` from the connected camera so that we can use this stream URL as an input when we are inferencing later. Execute the following command to start the TCP stream.

```python
rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
```

Learn more about `rpicam-vid` usage on [official Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-vid).

```python
from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Run inference
results = model("tcp://127.0.0.1:8888")

```


# Command to remember

```python
sudo apt update # updating raspberrypi - DO THIS ON DEFAULT EVERY TIME YOU INSTALL NEW LIBRARIES
sudo apt upgrade # upgrading raspberrypi
reboot
```




# Utilizing RaspberryPi AI HAT+

[TBA]






