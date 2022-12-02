from networks import ChainedModels
import glob
import timeit
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

import params as p
cntr = 0

def next_square_number(N):
    nextN = math.floor(math.sqrt(N)) + 1 
    return nextN * nextN

if __name__ == "__main__":
    start = timeit.default_timer()
    model = ChainedModels()
    print(f"Stand up time: {timeit.default_timer() - start}")

    if p.ENDLESS_MODE:
        while True:
            path = input("Give path to a picture: ")

            if path == "q":
                break
            
            start = timeit.default_timer()
            predictions = model.utilize_toolchain(path)
            end = timeit.default_timer() - start
            print(f"Prediction time: {end}")


            M = int(math.sqrt(next_square_number(len(predictions))))
            print(M)

            for i in range(len(predictions)):
                plt.subplot(M, M, i+1)
                plt.imshow(np.uint8(predictions[i][0]))
                plt.title(predictions[i][1])

            plt.savefig(f"./pred_{cntr}.png")
            cntr += 1
    else:
        paths = [f for f in glob.glob(fr"C:\Users\Nano\Downloads\MI\data\*")]
    
        predictions = []
        for f in paths:
            video_source = cv2.VideoCapture(f)
            success, frame = video_source.read()
            while success:
                predictions.append(model.utilize_toolchain(frame))
                success, frame = video_source.read()