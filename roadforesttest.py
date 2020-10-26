import pandas as pd
import houghtest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
import pickle

def main(img_path_or):
	trained_model=pickle.load(open("trained_model_25509.sav",'rb'))
	"""
	HEADERS = ["cn","b","g","r","bl","gl","rl","br","gr","rr","bu","gu","ru","bul","gul","rul",
		   "bur","gur","rur","bdl","gdl","rdl","bdr","gdr","rdr","bd","gd","rd","road"]
	print sorted(zip(trained_model.feature_importances_,HEADERS),reverse=True)
	"""
	candidates = [103,110,121,122,124,125,127,128,129,133,137,138,145,147,153,157,159,161,
			   164,165,167,170]
	for z in range(0,1):
		median_im = cv2.imread(img_path_or)
		img_copy = median_im.copy()

		#median_im = cv2.medianBlur(median_im, 3)
		#median_im = image
		for y in range(1, 318, 1):                            #4611 4778
			for c in range(1, 478, 1):
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
					pred = 1
				else:
					pred = 0
				if pred == 1:
					img_copy[y, c] = (255, 255, 0)
		cv2.imwrite("images/out1.png",img_copy)
		length = houghtest.main("images/out1.png",img_path_or)

		return length
	#cv2.imshow("pred", img_copy)
	#cv2.imwrite("pred8superr.jpg", img_copy)
	#cv2.waitKey(0)

if __name__=="__main__":
	main()
