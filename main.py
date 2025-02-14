from capture_image_module import capture_and_store
from process_module import process_new_images
import threading

if __name__ == "__main__":
    #capture_thread = threading.Thread(target=capture_and_store)
    process_thread = threading.Thread(target=process_new_images)

    #capture_thread.start()
    process_thread.start()

    #capture_thread.join()
    process_thread.join()