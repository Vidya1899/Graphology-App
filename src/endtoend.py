import sys
sys.path.insert(0, '/Users/yashbhat/GraphForLab')

import cv2
import sys
import glob
import sys
import pandas as pd
import numpy as np
import shutil
#import seaborn as sns
#import matplotlib.pyplot as plt
from statistics import mean, mode
from download.word_extract import extract_words
from download.baseline2 import baseline_extract
from download.baseline2 import get_size
# import statistics as stat
#from click_and_crop import get_ts
#from click_and_crop import get_hw
#from opencv_text_detection.text_detection import textdetection
from download.tbar import tbar_length
from download.hori import get_hori
from download.hori import get_f2
import time
import download.character_extract as ce
from keras.models import load_model
import os

def display_images(im):
	cv2.imshow("Image",im)
	cv2.waitKey(0)

def main(filename):
	# import picamera

	# with picamera.PiCamera() as camera:
	#     camera.resolution = (1024, 768)
	#     camera.start_preview()
	#     # Camera warm-up time
	#     time.sleep(10)
	#     camera.capture('foo.jpg')

	# foo = cv2.imread('foo.jpg')
	# get_hw(foo)

	# sample = cv2.imread('./HW_118_dataset/dataset-page-008.jpg')
	# sample = cv2.imread(sys.argv[1])
	#rpath = '/Users/yashbhat/Graphology/api/'
	# sample = cv2.imread(filename)
	sample = cv2.imread('./sample/'+filename)
	# height,width = sample.shape[0],sample.shape[1]
	# print("h,w:",height,width)
	# sample = cv2.resize(sample, dsize=(500, int(500 * height / width)), interpolation=cv2.INTER_AREA)

	# cv2.imshow('sample',sample)
	# cv2.waitKey(0)

	print("Handwriting Analysis...")
	#inp = input("Enter the /path/to/the image containing the handwriting:\n")
	#inputs = [cv2.imread(file) for file in glob.glob("../HW_118_dataset/dataset-page-008.jpg")]

	#img = cv2.imread(inp)

	#extract words

	#METHOD 1 : based on contour detection
	#extract_words(img)

	#METHOD 2: based on fast ease DL pre trained text detector
	#textdetection(input)
	#for input in inputs:

	start_time = time.time()
	extract_words(sample)
	#break

	# cv2.destroyAllWindows()

	# print("Load words to start analysing...")


	#read all words extracted from results
	# words = [cv2.imread(file) for file in glob.glob("./words/*")]
	words = [cv2.imread(file) for file in glob.glob('download/words/*')]

	# print("Select ts:")
	#words_t = [cv2.imread(file) for file in glob.glob("./words/*")]

	# for i,word_t in enumerate(words):
	# 	#t_present = False
	# 	#get_ts(word_t,i,t_present)
	# 	get_ts(word_t,i)

	console_display = []
	sizes = []

	for word in words:
		#find baseline
		try:
			baseline_img,f1_size,height = baseline_extract(word)
			# display_images(baseline_img)
			#print("word shape:",word.shape)
			sizes.append(get_size(f1_size,height))
			#sizes.append(f1_size)
			#heights.append(height)
			#temp = (f1_size*100)/height
			#p.append(temp)
		except:
			print("no size for this")
	# print("FEATURE 1: SIZE and SLANT : Level of Socially Outgoing.")

	# console_display.append("FEATURE 1: SIZE and SLANT : Level of Socially Outgoing.")

	# print('wtyfwagdsize',sizes)
	# print(sizes)
	# try:
	# 	print("Size:",mode(sizes))
	# 	console_display.append(str(mode(sizes)))
	# except:
	# 	print("Size:",sizes[0])
	# 	console_display.append(str(sizes[0]))

	map_size = [0,0,0]
	get_the_size = {0:'medium',1:'high',2:'low'}

	# sizes = ['high','low','high','medium','low']

	#map_size -> [medium, high and low]
	for size in sizes:
		if size == "high":
			# map_size['high'] += 1
			map_size[1] += 1
		elif size == 'medium':
			# map_size['medium'] += 1
			map_size[0] += 1
		else:
			# map_size['low'] += 1
			map_size[2] += 1

	# print(map_size)
	mx = max(map_size)
	# print(mx)
	mx1 = map_size.index(mx)
	# print(mx1)
	mx2 = get_the_size[mx1]
	# print(mx2)

	# console_display.append(str(sizes[0]))

	###ONE###
	console_display.append(mx2)
	# print(mx2)



	#read t cropped images
	#ts = [cv2.imread(file) for file in glob.glob("./ts/*")]

	#extract characters
	best_angle = ce.main(sample)
	# time1 = time.time()-start_time

	# print("put t in folders 't' ")
	# ts = [file for file in glob.glob("./ts/*")]
	characters = [cv2.imread(file,0) for file in glob.glob('download/characters/*')]
	# isitt = np.array(characters).reshape(len(characters), 28, 28, 1)
	isitt = np.array(characters).reshape(len(characters), 28, 28, 1)

	#detect 't' from characters
	model1 = load_model('download/model1.h5')
	preres = model1.predict(isitt)
	res = np.argmax(preres, axis=1)
	# print("Result for t :",res)

	# remove existing images
	dir_path = 'download/ts'
	try:
		shutil.rmtree(dir_path)
		os.mkdir(dir_path)
	except OSError as e:
		print("Error: %s : %s" % (dir_path, e.strerror))

	ts = []

	for i,r in enumerate(res):
		if r == 0:
			ts.append(characters[i])
			# print(preres[i])
			cv2.imwrite('download/ts/'+'ts'+str(i)+'.jpg',characters[i])

	if len(ts) < 1:
		console_display.append('limited')
		console_display.append('medium')
		console_display.append('high')
		console_display.append('medium')
		return console_display


	#
	# #find t bar length
	tbar_len = []
	tbar_pos = []
	opt = []
	# pos,neg,f3 = 0,0,0
	# optimism = {0: 'optimistic', 1: 'pessimistic', 99: 'undetermined'}
	# optimism = {0: 'high', 1: 'low', 99: 'undetermined'}
	#
	#
	for t in ts:
		try:
			#tlen, tangle = tbar_length(t)
			temp1,temp2,temp3 = get_hori(t)
			tbar_len.append(temp1)
			tbar_pos.append(temp2)
			opt.append(temp3)
			# print(temp1,temp2,temp3)
		except:
			print("invalid t image")

	# if pos < neg:
	# 	f3 = 1

	# print("FEATURE 2: Hook in the of t-bar : Degree of Tolerance.")
	# # console_display.append("FEATURE 2: Hook in the of t-bar : Degree of Tolerance.")
	# print(mean(tbar_len))
	# console_display.append(mean(tbar_len))

	try:
		f3 = mode(tbar_pos)
	except:
		f3 = tbar_pos[0]

	# f5 = get_f2(mean(tbar_len))
	try:
		f5 = mode(tbar_len)
	except:
		f5 = tbar_len[0]

	f6 = mode(opt)
	print('opt,f6',opt,f6)

	#t bar - bent - degree of tolerance
	###TWO###
	if (f6 == 99):
		console_display.append('limited')
	elif f6 == 0:
		console_display.append('limited')
	else:
		console_display.append('high')

	# t bar - position - practicality
	###THREE###
	console_display.append(f3)

	#t bar - angle / inclination - optimism
	###FOUR###
	if best_angle < 0:
		console_display.append('high')
	else:
		console_display.append('low')

	# t bar - length - enthusiasm
	###FIVE###
	console_display.append(f5)

	#
	#
	# if (f6 == 99):
	# 	print('High')
	# 	# console_display.append('High degree of tolerance')
	# else:
	# 	print('Limited')
	# 	# console_display.append('Limited degree of tolerance')
	#
	# print("FEATURE 3: Placement of t-bar : Practicality.")
	# # console_display.append("FEATURE 3: Placement of t-bar : Practicality.")
	# print(f3)
	# console_display.append(str(f3))
	#
	# print("FEATURE 4: The slant of the t-bar : Optimism.")
	# # console_display.append("FEATURE 4: The slant of the t-bar : Optimism.")
	# print(optimism[f6])
	# console_display.append(str(optimism[f6]))
	#
	# print("FEATURE 5: Length of t-bar : Enthusiasm.")
	# print(f5)
	# # console_display.append("FEATURE 5: Length of t-bar : Enthusiasm.")
	# console_display.append(str(f5))

	# print("total time ---- %s ----",time.time()-start_time)
	console_display.append("total time ---- %s ----" + str(time.time()-start_time))
	return console_display


if __name__ == "__main__":
	res = main(sys.argv[1])
	print(res)
