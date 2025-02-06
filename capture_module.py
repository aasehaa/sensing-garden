from picamera2 import Picamera2
import os
import time

'''
using Picamera2 for image capture and storage
'''

def capture_and_store():
    # Initialize the camera
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
    picam2.set_controls({"AfMode": 0})

    # if you want to configure the focus
    #picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})

    picam2.configure(camera_config)
    picam2.start()

    os.makedirs("frames", exist_ok=True)
    
    while True:
        image = picam2.capture_array()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"frames/image_{timestamp}.jpg"
        
        picam2.capture_file(filename)
        time.sleep(2)  # Capture every 10 seconds
