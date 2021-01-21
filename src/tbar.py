import numpy as np
import cv2
import sys
import math


def tbar_length(timage):

	#img = cv2.imread(timage_path)
	orig = timage.copy()
	gray = cv2.cvtColor(timage,cv2.COLOR_BGR2GRAY)
	#edges = cv2.Canny(gray,50,150,apertureSize = 3)
	edges = cv2.Canny(gray,50,150)
	#cv2.imshow('edges',edges)
	#cv2.waitKey()

	#arguments - image, P in pixels, theta in radians, votes, None, min length of line, gap allowed between to be a line
	#tested on approx 60x60 pixels
	#lines = cv2.HoughLinesP(edges,1,np.pi/180,15,None,1,10)
	lines = cv2.HoughLinesP(edges,1,np.pi/180,10,None,1,10)
	#print(lines)

	dist = []
	print("no of lines:",len(lines))
	#find longest of the lines found
	for i,line in enumerate(lines):
		x1 = line[0][0] 
		y1 = line[0][1]
		x2 = line[0][2]
		y2 = line[0][3]
		dist.append(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
		cv2.line(orig,(x1,y1),(x2,y2),(255,255,0),2)
		cv2.imshow('tbar-'+str(i+1),orig)
		cv2.waitKey(0)
		cv2.destroyWindow('tbar-'+str(i+1))

	which_line = input("Select the appropriate line for t bar:\n Choices are 1,2,3... depending on the line number\n")

	#idx = dist.index(max(dist))
	idx = int(which_line)-1
	#print(idx)

	deltaY = y2 - y1
	deltaX = x2 - x1

	angleInDegrees = math.atan2(deltaY, deltaX) * 180 / np.pi
	print(angleInDegrees)

	return (dist[idx],angleInDegrees)

if __name__ == '__main__':
	timage = cv2.imread(sys.argv[1])
	tlen = tbar_length(timage)
	print("tlength :",tlen)