import pathlib as pth
import os

WEIGHTS_NAME = "yolov4.weights"
CLASSES_NAME = "coco_classes.txt"

WORKING_DIR = pth.Path(os.getcwd())

CAR_DETECTION_PATH = WORKING_DIR.joinpath("car_detection")
TIRE_DETECTION_PATH = WORKING_DIR.joinpath("tire_detection")
TIRE_CLASSIFICATION_PATH = WORKING_DIR.joinpath("tire_classification")

TIRE_CLASSES = ["flat", "full", "⊙﹏⊙∥"]
CONFIDENCE_THRESHOLD = 0.7

SAVE_OP = True

ENDLESS_MODE = False
######################################################################
