#This script transfer the raw data of the test doodle images to real jpg images.

import os
from glob import glob
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

def get_images():
    fnames = 'test_simplified.csv'
    cnames = ['drawing', 'key_id']
    
    first = pd.read_csv(fnames) 
    examples = [ast.literal_eval(pts) for pts in first.drawing.values]#get every strokes in every images
    try:
        os.mkdir('test_images')
    except FileExistsError:
        print("Directory already exists")
    for i in range(len(examples)):
        for x,y in examples[i]:
            plt.gca().invert_yaxis()
            plt.plot(x, y, color=(0,0,0),marker='.',markersize=1,lw=1)#draw a stroke
            plt.axis('off')
        plt.savefig("{}.jpg".format(first['key_id'][i]))#save an image
        plt.clf()

#multiprocessing
if __name__ == '__main__':
    start = time.time()
    pool = mp.Pool(processes=24)
    pool.apply_async(get_images,)
    pool.close()
    pool.join()
    end = time.time()
    print(end-start)

