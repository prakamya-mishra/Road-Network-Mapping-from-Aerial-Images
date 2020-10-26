import pandas as pd
import houghtest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
import pickle
from joblib import Parallel, delayed
import multiprocessing
import time

img_path_or = 'test5.jpg'
median_im = cv2.imread(img_path_or)
img_copy = median_im.copy()

h, w = median_im.shape[:2]

test = 0



def main(img_path_or):

    global y, c, img_copy, h, w, trained_model
    #Parallel(n_jobs =  -1)(delayed(interprocess)(y) for y in range(h-1))
    #Parallel(n_jobs = -1, backend = 'threading')(delayed(process)(y, c) for y in range(h-1) for c in range(w-1))
    #Parallel(n_jobs = -1, backend = 'threading')(delayed(process)(c) for c in range(w-1))
    start = time.time()
    trained_model = pickle.load(open("trained_model_25509.sav",'rb'))
    #for y in range(h-1):
        #interprocess(y)
    for y in range(1, 319, 1):
        for c in range(1, 479, 1):
            process(y, c)
    length = houghtest.main("out1.png",img_path_or)
    end = time.time()
    print(end - start)
    return length
"""
def interprocess(y):
    global w
    Parallel(n_jobs = -1)(delayed(process)(y, c) for c in range(w-1))
"""
def process(y, c):
    global img_copy

    global trained_model
    b = median_im.item(y, c, 0)
    g = median_im.item(y, c, 1)
    r = median_im.item(y, c, 2)
    bl = median_im.item(y, c - 1, 0)
    gl = median_im.item(y, c - 1, 1)
    rl = median_im.item(y, c - 1, 2)
    br = median_im.item(y, c + 1, 0)
    gr = median_im.item(y, c + 1, 1)
    rr = median_im.item(y, c + 1, 2)
    bu = median_im.item(y - 1, c, 0)
    gu = median_im.item(y - 1, c, 1)
    ru = median_im.item(y - 1, c, 2)
    bul = median_im.item(y - 1, c - 1, 0)
    gul = median_im.item(y - 1, c - 1, 1)
    rul = median_im.item(y - 1, c - 1, 2)
    bur = median_im.item(y - 1, c + 1, 0)
    gur = median_im.item(y - 1, c + 1, 1)
    rur = median_im.item(y - 1, c + 1, 2)
    bdl = median_im.item(y + 1, c - 1, 0)
    gdl = median_im.item(y + 1, c - 1, 1)
    rdl = median_im.item(y + 1, c - 1, 2)
    bdr = median_im.item(y + 1, c + 1, 0)
    gdr = median_im.item(y + 1, c + 1, 1)
    rdr = median_im.item(y + 1, c + 1, 2)
    bd = median_im.item(y + 1, c, 0)
    gd = median_im.item(y + 1, c, 1)
    rd = median_im.item(y + 1, c, 2)

    new_prediction = trained_model.predict(np.array([[b, g, r, bl, gl, rl, br, gr, rr, bu, gu, ru, bul, gul, rul, bur, gur, rur, bdl, gdl, rdl, bdr, gdr, rdr, bd, gd, rd]]))
    if new_prediction > 0.5:
        img_copy[y, c] = (255, 255, 0)
    if y == 318 and c == 478:
        print('fuck')
        cv2.imwrite("out1.png",img_copy)



if __name__=="__main__":
    #print(median_im.shape(0))
    #print(median_im.shape(1))
    main(img_path_or)
