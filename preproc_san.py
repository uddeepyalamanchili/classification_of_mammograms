import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os
from math import sqrt

def histeq(img):
	img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
	hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
	return hist_equalization_result


def proc(img1):
	#img1 = cv2.imread('/home/nandini/Calcifications/all-mias/mdb219c.png')
	cv2.imwrite('/home/tinku/myfiles/final_project/exec_1/org.png',img1)
	img = cv2.medianBlur(img1, 3)
	#cv2.imshow("After median filtering",img);
	#cv2.waitKey(0)
	cv2.imwrite('/home/tinku/myfiles/final_project/exec_1/med.png',img)
	equ = histeq(img)
	#cv2.imshow("Image after HE",equ)
	#cv2.waitKey(0)
	psy = 0.8
	g1 = (img+psy*equ)/(1+psy)
	#cv2.imshow("Modified histogram image",g1)
	#cv2.waitKey(0)
	img = cv2.convertScaleAbs(g1)
	gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0)
	unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
	#cv2.imshow("After lce",unsharp_image)
	#cv2.waitKey(0)
	kernel = np.ones((15,15), np.uint8)
	img_dilation = cv2.dilate(unsharp_image, kernel, iterations=1)
	#cv2.imshow('Dilation', img_dilation)
	#cv2.waitKey(0)
	return img_dilation


def otsu(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	hist_norm = hist.ravel()/hist.max()
	Q = hist_norm.cumsum()
	bins = np.arange(256)
	fn_min = np.inf
	thresh = -1
	for i in xrange(1,256):
		p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
		q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
		if q1 == 0:
 			q1 = 0.00000001
		if q2 == 0:
# 		if q2 == 0:
 			q2 = 0.00000001
		b1,b2 = np.hsplit(bins,[i]) # weights
 		# finding means and variances
		m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
		v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
 		# calculates the minimization function
		fn = v1*q1 + v2*q2
		if fn < fn_min:
			fn_min = fn
			thresh = i
	ret2, otsu1 = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)
	return otsu1


def proc_tophat(img):
	kernel1 = np.ones((10,15),np.uint8)
	tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel1)
	return tophat

def clustering(img):
	Z = img.reshape((-1,3))
	# convert to np.float32
	Z = np.float32(Z)
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 8
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	cv2.imwrite('/home/tinku/myfiles/final_project/exec_1/clust.png',res2)
	#cv2.imshow('res2',res2)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return res2



def feat_ext(img_name,img):
	p = cv2.calcHist([img],[0],None,[256],[0,256])
	p_norm = p/(1024*1024);

	#mean = img.mean()


	mean = 0
	for z in range(1, 256):
		mean = mean+(z-1)*p_norm[z]
	mean = mean[0]

	var = 0
	for z in range(1,256):
		var = var + ( ( (z-1) - mean)*( (z-1) - mean) ) * p_norm[z];
	var = var[0]

	std = sqrt(var)
	R = 1 - (1/(1+ var ));

	skew = 0
	for z in range(1,256):
		skew = skew + ( ( ( (z-1) - mean)*( (z-1) - mean)*( (z-1) - mean)) * p_norm[z] );
	U = 0
	for z in range(1,256):
		U = U + (p_norm[z]*p_norm[z])
	print mean
	print var
	print std
	print R
	print skew[0]
	print U[0]
	feat = []

	feat.append(img_name)
	feat.append(mean)
	feat.append(var)
	feat.append(std)
	feat.append(R)
	feat.append(skew[0])
	feat.append(U[0])
	feat.append("\n")
	
	print feat
	feat = " ".join(str(x) for x in feat)
	print feat
	f = open('features.txt','a')
	f.write(feat)
	f.close()

	

	


"""
f = open('data.txt','r')
print f
while 1:
	img_name = f.readline()
	img_name = img_name.strip('\n')
	print img_name
	if len(img_name)!=0:
		path = '/home/osboxes/NEW/all-mias/'
		path = os.path.join(path,img_name)
		img = cv2.imread(path)
		
		pre_img = proc(img)
		#cv2.imshow("Preprocessed Image",pre_img)
		#cv2.waitKey(0)

		tophat = proc_tophat(pre_img)
		#cv2.imshow("TOPHAT",tophat)
		#cv2.waitKey(0)

		thresh = otsu(tophat)
		#cv2.imshow("Thresholded Image", thresh)
		#cv2.waitKey(0)
		cls=clustering(thresh)
		feat_ext(cls)

		

	else :
		break


"""
img_name = sys.argv[1]
print(type(img_name))
img_path = os.path.join('/home/tinku/myfiles/final_project/review_2/NEW_(copy)/all-mias',img_name)
print(img_path)
img = cv2.imread(img_path)

pre_img = proc(img)
cv2.imshow("Preprocessed Image",pre_img)
cv2.waitKey(0)

tophat = proc_tophat(pre_img)
cv2.imshow("TOPHAT",tophat)
cv2.waitKey(0)

thresh = otsu(tophat)
cv2.imshow("Thresholded Image", thresh)
cv2.waitKey(0)

cls=clustering(thresh)
cv2.imshow("clustered image", cls)
cv2.waitKey(0)

feat_ext(img_name,cls)






