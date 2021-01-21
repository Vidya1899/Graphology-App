import numpy as np
import cv2
import sys
# import itertools
import pandas as pd
# import statistics as stat


def count_through_a_list(x):
	"""
	returns all distinct continuous groups of values in a list
	output is in the form of records
	"""
	# Initialize these values
	group_start = 0
	group_count = 1
	prev = x[0]
	groups = []

	for i, n in enumerate(x):
		# if i == group_start:
		#   prev = n
		#   group_start = i
		if n != prev or i == len(x) - 1:
			groups.append({'start': group_start, 'end': i - 1, 'value': prev, 'length': i - group_start,
						   'group_counter': group_count})
			# Reset the appropriate values
			group_count += 1
			group_start = i
			prev = n
	# print(groups)
	return groups

def baseline_extract(image):
	#import image
	#image = cv2.imread(sys.argv[1])

	#grayscale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('gray',gray)
	#cv2.waitKey(0)

	#binary
        #150 , 250 -> 110,240
	ret,thresh = cv2.threshold(gray,110,240,cv2.THRESH_BINARY_INV)
	#cv2.imshow('thresh',thresh)
	#print(len(thHyh))
	#cv2.waitKey(0)

	Ix,Iy = thresh.shape
	#print('Ix,Iy',Ix,Iy)

	#print(thHyh[15:25,:])

	#Hy = []
	#WIDTH = img.shape[1]
	#HEIGHT = img.shape[0]

	#length of total black runs in the lines
	#equals Iy since contours encaptuHy whole word , so no empty white lines
	By = np.zeros(Iy)

	val = dict.fromkeys(range(Ix))
	#print(val)

	for row in range(Ix):
		val[row] = []
		start = 0
		for col in range(Iy):
			if(thresh[row,col] == 0) and (start == 0):
				start = col
			elif (not(thresh[row,col] == 0)) and (not(start == 0)):
				#wont do col-1 as adjusts with start which is inlucisve when we do end-start
				end = col
				val[row].append(end-start)
				start = 0
			else:
				pass

	#print(val)

	#Hy
	Hy = np.zeros(Ix)
	for k,v in val.items():
		Hy[k] += (len(v)**2) *  sum(v)

	#print(Hy)

	#threshold for core-region calculation
	#Th = (0.20/Iy)*sum(Hy)
	Th = (0.25 / Iy) * sum(Hy)

	#print(Th)

	HBy = np.zeros(Hy.shape)
	for row in range(Ix):
		if Hy[row] > Th:
			HBy[row] = 1

	#print(HBy)

	#print(argmax(HBy[1,:]))

	#print([list(group) for _, group in itertools.groupby(HBy)])

	groups = count_through_a_list(HBy)

	df = pd.DataFrame(groups, columns=['start','end','value', 'length', 'group_counter'])

	df2 = df[df['value'] == 1]
	#print(df2)
	max_index = df2[df2['length'] == max((df2['length']))]
	#print((max_index))

	#in case we encounter same max lengths
	max_index = max_index.iloc[0]
	#print("maxindex:")
	#print(max_index)
	# try:
	# 	#print(max_index)
	# 	pass
	# except:
	# 	pass
	#print(pd.DataFrame(groups, columns=['start','end','value', 'length', 'group_counter']))

	cv2.line(image,(0,int(max_index['start'])),(Iy,int(max_index['start'])),(255,20,147),1)
	cv2.line(image,(0,int(max_index['end'])),(Iy,int(max_index['end'])),(255,20,147),1)
	# cv2.imshow('Image Baseline',image)
	# cv2.waitKey(0)

	return image,int(max_index['length']),int(max_index['end'])

def get_size(size,height):
	temp = (size*100)/height
	#print(temp)
	res = ''
	if (temp >= 70):
		res = 'high'
	elif (temp >= 50):
		res = 'medium'
	else:
		res = 'low'
	return res

if __name__ == "__main__":
	image = cv2.imread(sys.argv[1])
	baseline_extract(image)
