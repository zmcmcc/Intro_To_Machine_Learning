#This script transfer the raw data of the test doodle images to real jpg images.

import os
from glob import glob
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

def get_images(start,end):#340 classes
    fnames = glob('../train_simplified/*.csv')#get all 340 class names
    cnames = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
    
    for f in range(start,end):
        drawlist = []
        first = pd.read_csv(fnames[f]) 
        first['word'] = first['word'].replace(' ', '_', regex=True)
        first = first[first.recognized==True][:20000]# make sure we get 20000 recognized drawings
        drawlist.append(first)
        draw_df = pd.DataFrame(np.concatenate(drawlist), columns=cnames)
        examples = [ast.literal_eval(pts) for pts in draw_df.drawing.values]#get every strokes in every images
        dirName = first['word'][0]
        try:#create folders for every classes
            os.mkdir(dirName)
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")
        for i in range(len(examples)):
            for x,y in examples[i]:
                plt.gca().invert_yaxis()
                plt.plot(x, y, color=(0,0,0),marker='.',markersize=1,lw=1)#draw a stroke
                plt.axis('off')
            plt.savefig("{}/{}_{}.jpg".format(dirName,dirName,i))#save an image
            plt.clf()

#multiprocessing
if __name__ == '__main__':
    start = time.time()
    pool = mp.Pool(processes=24)
    for i in range(252,253):
        pool.apply_async(get_images,(i,i+1,))
    pool.close()
    pool.join()
    end = time.time()
    print(end-start)

