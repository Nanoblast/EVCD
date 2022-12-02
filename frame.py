from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import pathlib as pth
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL as pil
import numpy as np
import cv2
from glob import glob
import sys
from timeit import default_timer as timer
import torch


# Set paths relative to git default directory
CURRENT_PATH = pth.Path(os.getcwd())
YOLO_PATH = CURRENT_PATH.joinpath("Framework/yolo/")
sys.path.append(YOLO_PATH.as_posix())

from yolo.models import Yolov4

#----------------------------------
#Global variables come here
#----------------------------------
WEIGHTS_NAME = "yolov4.weights"
CLASSES_NAME = "coco_classes.txt"
TIRE_CLASSIFICATION_MODEL_PATH = CURRENT_PATH.joinpath("Framework/tire_classification/20220617-173955")
TIRE_CLASSIFICATION_MODEL_URL = "https://drive.google.com/drive/folders/13LMNPsqdgTt7d7sJLeyows0fRi96d-DC?usp=sharing"
TIRE_CLASSES = ["flat", "full"]

#----------------------------------
#Functions come here
#----------------------------------
def read_image_from_path(path : str) -> np.ndarray:
    image = tf.keras.preprocessing.image.load_img(fr"{path}", target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    return input_arr

def interpret_tire_classification_output(distribution):
    return TIRE_CLASSES[np.argmax(distribution)]

def get_drive_model(url, destination):
    pass

#----------------------------------
#Global models
#----------------------------------

#  Base car detection model TODO: change this model to yolov5
print("LOAD YOLO BASE MODEL")
yolo_v4_model = Yolov4(weight_path=YOLO_PATH.joinpath(WEIGHTS_NAME).as_posix(),
                    class_name_path=YOLO_PATH.joinpath(CLASSES_NAME).as_posix())

# Tire detection model
print("LOAD YOLO TIRE DETECTION MODEL")
yolo_v5_model = torch.hub.load(r'C:\Users\ktama\smart_light\code\Sandbox\yolo_v5_tire\env\yolov5',
                            'custom',
                            r'C:\Users\ktama\smart_light\code\Framework\tire_detection\best.pt',
                            source='local',)

#  Tire classification model
print("LOAD TIRE CLASSIFICATION MODEL")
tire_classification_model = tf.keras.models.load_model(TIRE_CLASSIFICATION_MODEL_PATH.as_posix())
    

#Handler for incomming requests.
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        #------------------------------
        #Call evaluation function here()
        #------------------------------


        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write(bytes("OK", "utf8"))
        #TBD: return evaluated data in response!

if __name__ == "__main__":
    #------------------------------
    #Initialize ML paramteres here!
    #------------------------------

    # EXAMPLE USAGES #
    path = CURRENT_PATH.joinpath("DemoData/flat.png").as_posix()  # CURRENT_PATH.joinpath("DemoData/full.png").as_posix()
    preproc_image = read_image_from_path(path)
    predicted_distribution = tire_classification_model.predict(preproc_image)
    predicted_label = interpret_tire_classification_output(predicted_distribution)   
    print(predicted_label) # ["flat", "full"] 

    path = CURRENT_PATH.joinpath("DemoData/yolo_v4_demo_01.jpg").as_posix()  # CURRENT_PATH.joinpath("DemoData/yolo_v4_demo_02.jpg").as_posix()
    car_images = yolo_model.predict_only_cars(path)
    print(f"Found card bounding boxes: {len(car_images)}")

    #for img in car_images:
    #    plt.imshow(img)
    #    plt.show()

    #Hardwired adress and port, can be parameterized later if needed.
    print("Started receiving mode")
    with HTTPServer(("localhost", 12044), handler) as server:
        server.serve_forever()