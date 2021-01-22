import sys
sys.path.append("./src/")
import cv2
import sys
import glob
import sys
import pandas as pd
import numpy as np
import shutil
from statistics import mean, mode
from word_extract import extract_words
from baseline2 import baseline_extract
from baseline2 import get_size
from hori import get_hori
from hori import get_f2
import time
import character_extract as ce
from keras.models import load_model
import os

def display_images(im):
	cv2.imshow("Image",im)
	cv2.waitKey(0)

def runInference(filename):
	sample = cv2.imread(filename)
	print("Handwriting Analysis...")
	extract_words(sample)
	words = [cv2.imread(file) for file in glob.glob('samples/*')]
	console_display = []
	sizes = []

	for word in words:
		#find baseline
		try:
			baseline_img,f1_size,height = baseline_extract(word)
		except:
			print("no size for this")
	map_size = [0,0,0]
	get_the_size = {0:'medium',1:'high',2:'low'}
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
	characters = [cv2.imread(file,0) for file in glob.glob('samples/characters/*')]
	# isitt = np.array(characters).reshape(len(characters), 28, 28, 1)
	isitt = np.array(characters).reshape(len(characters), 28, 28, 1)

	#detect 't' from characters
	model1 = load_model('models/model1.h5')
	preres = model1.predict(isitt)
	res = np.argmax(preres, axis=1)
	# print("Result for t :",res)

	# remove existing images
	dir_path = 'samples/ts'
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
			cv2.imwrite('samples/ts/'+'ts'+str(i)+'.jpg',characters[i])

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
	#console_display.append("total time ---- %s ----" + str(time.time()-start_time))
	return console_display

