#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from libxmp import XMPFiles

def main(img_path_plotted,img_path_or):
	cc=0
	labels=np.zeros((320,480),dtype='int')
	candidates = [103,110,121,122,124,125,127,128,129,133,137,138,145,147,153,157,159,161,
			   164,165,167,170]
	for qq in range(0,1,1):
		labels=np.zeros((320,480),dtype='int')
		median_im_or = cv2.imread(img_path_or)
		median_im = cv2.imread(img_path_plotted)
		xmpfile = XMPFiles(file_path = img_path_or, open_forupdate = False)

		xmp = xmpfile.get_xmp()

		altitude = xmp.get_property("http://www.dji.com/drone-dji/1.0/", 'RelativeAltitude')
		altitude = float(altitude)
		altitude = altitude*100;

		xmpfile.close_file()

		img_width = 480
		img_height = 320
		sensorw = 1.32
		sensorh = 0.88
		focal = 0.88
		GSDW = (altitude*sensorw)/(focal*img_width)
		GSDH = (altitude*sensorh)/(focal*img_height)

		GSD = max(GSDW, GSDH)
		img_copy = median_im.copy()
		x_parent={}
		count={}
		#pass1
		for y in range(1,319,1):
			for c in range(1,479,1):
				b = median_im.item(y, c, 0)
				g = median_im.item(y, c, 1)
				r = median_im.item(y, c, 2)
				if (b==255 and g==255 and r==0):

					if (labels[y,c-1]==0 and labels[y-1,c]==0):
						cc = cc + 1
						labels[y,c]=cc
						x_parent[cc]=cc
					elif (labels[y,c-1]==0 and labels[y-1,c]!=0):
						labels[y,c]=labels[y-1,c]
					elif (labels[y,c-1]!=0 and labels[y-1,c]==0):
						labels[y,c]=labels[y,c-1]
					elif (labels[y,c-1]!=0 and labels[y-1,c]!=0):
						if (labels[y,c-1]==labels[y-1,c]):
							labels[y,c]=labels[y-1,c]
						elif (labels[y,c-1]>labels[y-1,c]):
							labels[y,c]=labels[y-1,c]
							x_parent[labels[y,c-1]]=labels[y-1,c]
						elif (labels[y,c-1]<labels[y-1,c]):
							labels[y,c]=labels[y,c-1]
							x_parent[labels[y-1,c]]=labels[y,c-1]
		#pass2
		for i in range(1,319,1):
			for j in range(1,479,1):
				temp = labels[i,j]
				if (temp==0):
					continue
				elif (temp==x_parent[temp]):
					if (temp in count.keys()):
						count[temp]=count[temp]+1
					else:
						count[temp]=1
				elif (temp!=x_parent[temp]):
					tempp=find(temp,x_parent)
					if (tempp in count.keys()):
						count[tempp]=count[tempp]+1
					else:
						count[tempp]=1
					labels[i,j]=tempp

		#operating
		threshold=90
		defected = []
		for key in count:
			if (count[key]<threshold):
				defected.append(key)


		for i in range(1,319,1):
			for j in range(1,479,1):
				if (labels[i,j] in defected):
					labels[i,j]=0
					b_temp=median_im_or.item(i,j,0)
					g_temp=median_im_or.item(i,j,1)
					r_temp=median_im_or.item(i,j,2)
					img_copy[i,j] = (b_temp,g_temp,r_temp)

		#houghlpotting
		labels_list = []
		finalpixels = []
		for i in range(0,320,1):
			for j in range(0,480,1):
				if (labels[i,j] in labels_list):
					continue
				else:
					labels_list.append(labels[i,j])
		for l in labels_list:
			if (l==0):
				continue
			else:
				reqind = 0
				reqlen = 0
				houghpixels = []
				maxpixels = []
				img_temp_copy = img_copy.copy()
				for i in range(0,320,1):
					for j in range(0,480,1):
						if (labels[i,j]!=l):
							img_temp_copy[i,j]=(0,0,0)
				img_temp_or = img_temp_copy.copy()
				gray = cv2.cvtColor(img_temp_copy,cv2.COLOR_BGR2GRAY)
				edges = cv2.Canny(gray,500,700,apertureSize = 3)
				lines = cv2.HoughLines(edges,1,np.pi/180,10)
				try:
					for line in lines:
						rho,theta = line[0]
						a = np.cos(theta)
						b = np.sin(theta)
						x0 = a*rho
						y0 = b*rho
						x1 = int(x0 + 1000*(-b))
						y1 = int(y0 + 1000*(a))
						x2 = int(x0 - 1000*(-b))
						y2 = int(y0 - 1000*(a))
						cv2.line(img_temp_copy,(x1,y1),(x2,y2),(0,0,255),1)
						for i in range(0,320,1):
							for j in range(0,480,1):
								b = img_temp_copy.item(i,j,0)
								g = img_temp_copy.item(i,j,1)
								r = img_temp_copy.item(i,j,2)
								if (b==0 and g==0 and r==255):
									b_or = img_temp_or.item(i,j,0)
									g_or = img_temp_or.item(i,j,1)
									r_or = img_temp_or.item(i,j,2)
									if (b_or==255 and g_or==255 and r_or==0):
										houghpixels.append([i,j])
										continue
						maxpixels.append(houghpixels)
						img_temp_copy = img_temp_or.copy()
						houghpixels = []
				except TypeError:
					continue

				for p in maxpixels:
					if (len(p)>reqlen):
						reqlen = len(p)
						reqind = maxpixels.index(p)
				finalpixels.append(maxpixels[reqind])
		pixelcount = 0
		for pixel in finalpixels:
			for i in pixel:
				pixelcount = pixelcount + 1
				img_copy[i[0],i[1]]=(0,0,255)

		length = pixelcount*GSD/100
		print(length)
		cv2.imwrite("images/out2.png",img_copy)

		return length


def find(l,mydict={}):
	if (mydict[l]==l):
		return l
	else:
		return find(mydict[l],mydict)

if __name__=="__main__":
	main()
