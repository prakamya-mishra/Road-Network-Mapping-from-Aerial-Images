import pandas as pd
import houghtest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
import pickle
from multiprocessing import Process
import time

def thread1():
    global h, w, trained_model, copy1
    newcopy1 = copy1.copy()
    for y in range((h/2)-1):
        for c in range((w/2)-1):
            b = newcopy1.item(y, c, 0)
            g = newcopy1.item(y, c, 1)
            r = newcopy1.item(y, c, 2)
            bl = newcopy1.item(y, c - 1, 0)
            gl = newcopy1.item(y, c - 1, 1)
            rl = newcopy1.item(y, c - 1, 2)
            br = newcopy1.item(y, c + 1, 0)
            gr = newcopy1.item(y, c + 1, 1)
            rr = newcopy1.item(y, c + 1, 2)
            bu = newcopy1.item(y - 1, c, 0)
            gu = newcopy1.item(y - 1, c, 1)
            ru = newcopy1.item(y - 1, c, 2)
            bul = newcopy1.item(y - 1, c - 1, 0)
            gul = newcopy1.item(y - 1, c - 1, 1)
            rul = newcopy1.item(y - 1, c - 1, 2)
            bur = newcopy1.item(y - 1, c + 1, 0)
            gur = newcopy1.item(y - 1, c + 1, 1)
            rur = newcopy1.item(y - 1, c + 1, 2)
            bdl = newcopy1.item(y + 1, c - 1, 0)
            gdl = newcopy1.item(y + 1, c - 1, 1)
            rdl = newcopy1.item(y + 1, c - 1, 2)
            bdr = newcopy1.item(y + 1, c + 1, 0)
            gdr = newcopy1.item(y + 1, c + 1, 1)
            rdr = newcopy1.item(y + 1, c + 1, 2)
            bd = newcopy1.item(y + 1, c, 0)
            gd = newcopy1.item(y + 1, c, 1)
            rd = newcopy1.item(y + 1, c, 2)

            new_prediction = trained_model.predict(np.array([[b, g, r, bl, gl, rl, br, gr, rr, bu, gu, ru, bul, gul, rul, bur, gur, rur, bdl, gdl, rdl, bdr, gdr, rdr, bd, gd, rd]]))
            if new_prediction > 0.5:
                copy1[y, c] = (255, 255, 0)
        cv2.imwrite("copy1.png",copy1)

def thread2():
    global h, w, trained_model, copy2
    newcopy2 = copy2.copy()
    for y in range((h/2)-1):
        for c in range((w/2)-1):
            b = newcopy2.item(y, c, 0)
            g = newcopy2.item(y, c, 1)
            r = newcopy2.item(y, c, 2)
            bl = newcopy2.item(y, c - 1, 0)
            gl = newcopy2.item(y, c - 1, 1)
            rl = newcopy2.item(y, c - 1, 2)
            br = newcopy2.item(y, c + 1, 0)
            gr = newcopy2.item(y, c + 1, 1)
            rr = newcopy2.item(y, c + 1, 2)
            bu = newcopy2.item(y - 1, c, 0)
            gu = newcopy2.item(y - 1, c, 1)
            ru = newcopy2.item(y - 1, c, 2)
            bul = newcopy2.item(y - 1, c - 1, 0)
            gul = newcopy2.item(y - 1, c - 1, 1)
            rul = newcopy2.item(y - 1, c - 1, 2)
            bur = newcopy2.item(y - 1, c + 1, 0)
            gur = newcopy2.item(y - 1, c + 1, 1)
            rur = newcopy2.item(y - 1, c + 1, 2)
            bdl = newcopy2.item(y + 1, c - 1, 0)
            gdl = newcopy2.item(y + 1, c - 1, 1)
            rdl = newcopy2.item(y + 1, c - 1, 2)
            bdr = newcopy2.item(y + 1, c + 1, 0)
            gdr = newcopy2.item(y + 1, c + 1, 1)
            rdr = newcopy2.item(y + 1, c + 1, 2)
            bd = newcopy2.item(y + 1, c, 0)
            gd = newcopy2.item(y + 1, c, 1)
            rd = newcopy2.item(y + 1, c, 2)

            new_prediction = trained_model.predict(np.array([[b, g, r, bl, gl, rl, br, gr, rr, bu, gu, ru, bul, gul, rul, bur, gur, rur, bdl, gdl, rdl, bdr, gdr, rdr, bd, gd, rd]]))
            if new_prediction > 0.5:
                copy2[y, c-(w/2)] = (255, 255, 0)
        cv2.imwrite("copy2.png", copy2)

def thread3():
    global h, w, trained_model, copy3
    newcopy3 = copy3.copy()
    for y in range((h/2)-1):
        for c in range((w/2)-1):
            b = newcopy3.item(y, c, 0)
            g = newcopy3.item(y, c, 1)
            r = newcopy3.item(y, c, 2)
            bl = newcopy3.item(y, c - 1, 0)
            gl = newcopy3.item(y, c - 1, 1)
            rl = newcopy3.item(y, c - 1, 2)
            br = newcopy3.item(y, c + 1, 0)
            gr = newcopy3.item(y, c + 1, 1)
            rr = newcopy3.item(y, c + 1, 2)
            bu = newcopy3.item(y - 1, c, 0)
            gu = newcopy3.item(y - 1, c, 1)
            ru = newcopy3.item(y - 1, c, 2)
            bul = newcopy3.item(y - 1, c - 1, 0)
            gul = newcopy3.item(y - 1, c - 1, 1)
            rul = newcopy3.item(y - 1, c - 1, 2)
            bur = newcopy3.item(y - 1, c + 1, 0)
            gur = newcopy3.item(y - 1, c + 1, 1)
            rur = newcopy3.item(y - 1, c + 1, 2)
            bdl = newcopy3.item(y + 1, c - 1, 0)
            gdl = newcopy3.item(y + 1, c - 1, 1)
            rdl = newcopy3.item(y + 1, c - 1, 2)
            bdr = newcopy3.item(y + 1, c + 1, 0)
            gdr = newcopy3.item(y + 1, c + 1, 1)
            rdr = newcopy3.item(y + 1, c + 1, 2)
            bd = newcopy3.item(y + 1, c, 0)
            gd = newcopy3.item(y + 1, c, 1)
            rd = newcopy3.item(y + 1, c, 2)

            new_prediction = trained_model.predict(np.array([[b, g, r, bl, gl, rl, br, gr, rr, bu, gu, ru, bul, gul, rul, bur, gur, rur, bdl, gdl, rdl, bdr, gdr, rdr, bd, gd, rd]]))
            if new_prediction > 0.5:
                copy3[y-(h/2), c] = (255, 255, 0)
        cv2.imwrite("copy3.png", copy3)

def thread4():
    global h, w, trained_model, copy4
    newcopy4 = copy4.copy()
    for y in range((h/2)-1):
        for c in range((w/2)-1):
            b = newcopy4.item(y, c, 0)
            g = newcopy4.item(y, c, 1)
            r = newcopy4.item(y, c, 2)
            bl = newcopy4.item(y, c - 1, 0)
            gl = newcopy4.item(y, c - 1, 1)
            rl = newcopy4.item(y, c - 1, 2)
            br = newcopy4.item(y, c + 1, 0)
            gr = newcopy4.item(y, c + 1, 1)
            rr = newcopy4.item(y, c + 1, 2)
            bu = newcopy4.item(y - 1, c, 0)
            gu = newcopy4.item(y - 1, c, 1)
            ru = newcopy4.item(y - 1, c, 2)
            bul = newcopy4.item(y - 1, c - 1, 0)
            gul = newcopy4.item(y - 1, c - 1, 1)
            rul = newcopy4.item(y - 1, c - 1, 2)
            bur = newcopy4.item(y - 1, c + 1, 0)
            gur = newcopy4.item(y - 1, c + 1, 1)
            rur = newcopy4.item(y - 1, c + 1, 2)
            bdl = newcopy4.item(y + 1, c - 1, 0)
            gdl = newcopy4.item(y + 1, c - 1, 1)
            rdl = newcopy4.item(y + 1, c - 1, 2)
            bdr = newcopy4.item(y + 1, c + 1, 0)
            gdr = newcopy4.item(y + 1, c + 1, 1)
            rdr = newcopy4.item(y + 1, c + 1, 2)
            bd = newcopy4.item(y + 1, c, 0)
            gd = newcopy4.item(y + 1, c, 1)
            rd = newcopy4.item(y + 1, c, 2)

            new_prediction = trained_model.predict(np.array([[b, g, r, bl, gl, rl, br, gr, rr, bu, gu, ru, bul, gul, rul, bur, gur, rur, bdl, gdl, rdl, bdr, gdr, rdr, bd, gd, rd]]))
            if new_prediction > 0.5:
                copy4[y-(h/2), c-(w/2)] = (255, 255, 0)
        cv2.imwrite("copy4.png", copy4)

def main(img_path_or):
    global trained_model, copy1, copy2, copy3, copy4, h, w

    start = time.time()

    print('Unpacking model')
    trained_model = pickle.load(open("trained_model_25509_wo_verbose.sav",'rb'))

    img = cv2.imread(img_path_or)
    h, w = img.shape[:2]

    copy1 = img[0:(h/2), 0:(w/2)]
    copy2 = img[0:(h/2), (w/2):w]
    copy3 = img[(h/2):h, 0:(w/2)]
    copy4 = img[(h/2):h, (w/2):w]

    print('Pocessing')
    p1 = Process(target=thread1)
    p2 = Process(target=thread2)
    p3 = Process(target=thread3)
    p4 = Process(target=thread4)

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    out1 = np.zeros((320, 480, 3))

    out1[0:(h/2), 0:(w/2)] = cv2.imread('copy1.png')
    out1[0:(h/2), (w/2):w] = cv2.imread('copy2.png')
    out1[(h/2):h, 0:(w/2)] = cv2.imread('copy3.png')
    out1[(h/2):h, (w/2):w] = cv2.imread('copy4.png')

    cv2.imwrite('images/out1.png', out1)

    length = houghtest.main("images/out1.png",img_path_or)
    print('finished')
    end = time.time()
    print('Took '+str(round(((end - start)/60), 2))+' mins to process')
    return length

if __name__ == '__main__':
    main(img_path_or)
