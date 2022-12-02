import datetime
import tensorflow as tf
import torch
import pathlib as pth
import numpy as np
from typing import Union, List, Tuple
import cv2
import os
import shutil
import timer
import json
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

bucket = "IOT"
org = "thesis"
token = "2fdtM9JCOYElGwOT2O367eL6j9PXjfhl59f4R04WBRTdHFMRPwETqqzwFYCUl-64u93xJpDxNcZosKzv8Sc_wQ=="
url="192.168.0.13:8086"

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)
write_api = client.write_api(write_options=SYNCHRONOUS)
cntr = 0

from car_detection.models import Yolov4
from tire_detection import *

import params as p
database = []
class ChainedModels():
    def __init__(self) -> None:
        # Car detector
        self.car_detector = Yolov4(weight_path=p.CAR_DETECTION_PATH.joinpath(p.WEIGHTS_NAME).as_posix(),
                                   class_name_path=p.CAR_DETECTION_PATH.joinpath(p.CLASSES_NAME).as_posix())

        # Tire detector
        self.tire_detector = torch.hub.load(r'C:\Users\Nano\Downloads\MI\car_condition_survey\tire_detection',
                            'custom',
                            r'C:\Users\Nano\Downloads\MI\car_condition_survey\tire_detection\best.pt',
                            source='local')

        # Tire classificator
        self.tire_classificator = tf.keras.models.load_model(p.TIRE_CLASSIFICATION_PATH.as_posix())

    def mobilize_car_detection(self, image):
        cars_list = self.car_detector.predict_only_cars(image)

        return cars_list

    def mobilize_tire_detection(self, image):
        result = self.tire_detector(image)

        return result.pandas().xyxy[0]

    def mobilize_tire_classification(self, image):
        predicted_distr = self.tire_classificator.predict(image)
        predicted_label = self.interpret_tire_classification_output(predicted_distr)

        return predicted_label

    def read_image_from_path(self, path : str) -> np.ndarray:
        image = tf.keras.preprocessing.image.load_img(fr"{path}")
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.

        return input_arr
    def read_video_from_path(self, path : str) -> np.ndarray:
        video_source = cv2.VideoCapture('path')
        success, frame = video_source.read()
        while success:
            image = tf.keras.preprocessing.image.load_img(frame)
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.

    def interpret_tire_classification_output(self, distribution):
        confident = np.max(distribution) > p.CONFIDENCE_THRESHOLD
        if confident:
            return p.TIRE_CLASSES[np.argmax(distribution)]
        else:
            return p.TIRE_CLASSES[-1]

    def utilize_toolchain(self, image : Union[str, np.ndarray]) -> Tuple[str, np.ndarray]:

        global cntr
        cntr = 0

        
        if type(image) == str:
            raw_image = self.read_image_from_path(image)
        if type(image) == np.ndarray:
            print()
            raw_image = image

        if p.SAVE_OP:
            date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            save_path = fr"C:\Users\Nano\Downloads\MI\car_condition_survey\pred\{date}_{np.random.randint(0, 10)}"                
            os.mkdir(save_path)
        #    shutil.copy(image, f"{save_path}\\")

        car_image_list = self.mobilize_car_detection(np.squeeze(raw_image))
        print(f"Number of cars: {len(car_image_list)}")

        tire_buffer_list = []
        for i, car in enumerate(car_image_list):
            dataline = ""
            pointer = ''
            if p.SAVE_OP:
                os.mkdir(fr"{save_path}\{cntr}")
                cv2.imwrite(f"{save_path}\{cntr}\car_{i}.png", car)
                dataline = str(f"{save_path}\{cntr}\car_{i}.png")
                

            result = self.mobilize_tire_detection(car)
            inncer_cntr = 0
            for _, row in result.iterrows():
                xmin = int(row["xmin"])
                xmax = int(row["xmax"])
                ymin = int(row["ymin"])
                ymax = int(row["ymax"])
                tmp = np.squeeze(car)[ymin:ymax, xmin:xmax, :]
                if p.SAVE_OP:
                    cv2.imwrite(f"{save_path}\{cntr}\\tire_{inncer_cntr}.png", tmp)
                try:
                    dataline +=  "," + str(self.mobilize_tire_classification(np.array([cv2.resize(tmp, (224, 224))])))
                    #dataline[f"{save_path}\{cntr}\\tire_{inncer_cntr}.png"] =  self.mobilize_tire_classification(np.array([cv2.resize(tmp, (224, 224))]))
                    result = self.mobilize_tire_classification(np.array([cv2.resize(tmp, (224, 224))]))
                    match(result):
                        case 'flat':
                            result = 0
                        case 'full':
                            result = 1
                        case _:
                            result = -1     

                    pointer = influxdb_client.Point("car_detection").tag("Picture", f"{save_path}\{cntr}\car_{i}.png").field(f"{save_path}\{cntr}\\tire_{inncer_cntr}.png", result)
                except:
                    #dataline[f"{save_path}\{cntr}\\tire_{inncer_cntr}.png"] = "n/a"
                    dataline += ",n/a"
                    pointer = influxdb_client.Point("car_detection").tag("Picture", f"{save_path}\{cntr}\car_{i}.png").field(f"{save_path}\{cntr}\\tire_{inncer_cntr}.png", result)
                tire_buffer_list.append(tmp)
                inncer_cntr += 1

            cntr += 1
            #write_api.write(bucket=bucket, org=org, record=pointer)
            with open(fr"C:\Users\Nano\Downloads\MI\car_condition_survey\pred\sum.csv", "a") as f:
                try:
                    print("write")
                    f.write(dataline)
                    f.write("\n")
                except:
                    print('missed')
                    f.write("Exemplaris Excomunicationis\n")
        
        print(f"Number of tires: {len(tire_buffer_list)}")

        result_buffer = []
        label_buffer = []
        for tire in tire_buffer_list:
            label = self.mobilize_tire_classification(np.array([cv2.resize(tire, (224, 224))]))
            result_buffer.append((tire, label))
            label_buffer.append(label)
        if p.SAVE_OP:
            with open(fr"C:\Users\Nano\Downloads\MI\car_condition_survey\pred\sum.csv", "a") as f:
                combined_list = [f"{k} : {v}" for k, v in zip([*range(len(label_buffer))], label_buffer)]
        return result_buffer
        
        