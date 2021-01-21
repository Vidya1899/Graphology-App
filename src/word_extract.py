import cv2
import numpy as np
import sys
import os
import shutil
# import matplotlib.pyplot as plt

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def display_images(im):
	cv2.imshow("Image",im)
	cv2.waitKey(0)


def extract_words(image):
	#cv2.imshow('orig',image)
	#cv2.waitKey(0)


	#uniform scale of input images
	# print("=======================================================================")

	dir_path = 'download/words'
	try:
		shutil.rmtree(dir_path)
		os.mkdir(dir_path)
	except OSError as e:
		print("Error: %s : %s" % (dir_path, e.strerror))


	#uniform scale ????
	draw_img = image.copy()
	w,h,d = image.shape
	#if (w > 500) or (h > 500):
		#image = cv2.resize(image, (500,500), interpolation = cv2.INTER_AREA)
		#image = cv2.resize(image, (800,800))


	#grayscale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# cv2.imshow('gray',gray)
	# cv2.waitKey(0)

	#binary
	#ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        #110,240 -> 120,240
	ret,thresh = cv2.threshold(gray,120,240,cv2.THRESH_BINARY_INV)
	# cv2.imshow('thresh',thresh)
	# cv2.waitKey(0)

	#dilation
	kernel = np.ones((5,5), np.uint8)
	img_dilation = cv2.dilate(thresh, kernel, iterations=3)
	# cv2.imshow('dilated',img_dilation)
	# cv2.waitKey(0)


	#find contours
	ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#sort contours
	#sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	sorted_ctrs = sorted(ctrs, key=lambda x: cv2.contourArea(x))
	new_ctr = []

	for c in sorted_ctrs:
	    #print(cv2.contourArea(c))
	    #for 500x500 or lesser dim vlue ball paointed to 250 or less minimum area
	    if(cv2.contourArea(c)>100):
	        new_ctr.append(c)

    #best 5 candidates
	if len(new_ctr) <= 5:
		print(len(new_ctr)+" of contours")
		pass
	else:
		new_ctr = new_ctr[-6:-1]

	#print(len(new_ctr))

	# f = plt.figure()
	# n = len(new_ctr)



	#for i, ctr in enumerate(sorted_ctrs):
	for i, ctr in enumerate(new_ctr):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)

		# Getting ROI
		roi = image[y:y+h, x:x+w]

		# show ROI
		# cv2.imshow('segment no:'+str(i),roi)
		cv2.imwrite(os.path.join('download/words/', 'res_' + str(i) + '.png'), roi)
		cv2.rectangle(draw_img,(x,y),( x + w, y + h ),(90,0,255),2)
		# f.add_subplot(1, n, i + 1)
		# plt.imshow(roi)
		# cv2.waitKey(0)

	#cv2.imshow('marked areas',draw_img)
	#cv2.waitKey(0)

	# print("Words extracted are stored in 'words' folder")

	# plt.show(block=True)

	return


if __name__ == '__main__':
	#import image
	image = cv2.imread(sys.argv[1])
	extract_words(image)
