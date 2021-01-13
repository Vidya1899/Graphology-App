# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:39:17 2020

@author: VIDYA
"""
import os, io
from google.cloud import vision
from google.cloud.vision_v1 import types

from spacy.lang.en import English
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()
import nltk
nltk.download('punkt')
from spellchecker import SpellChecker
spell = SpellChecker()
from nltk.tokenize import sent_tokenize 
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk.stem import PorterStemmer
porter = PorterStemmer()
lancaster=LancasterStemmer()

contentDictionary = {
    "Normal":"Hey there ! Your handwriting is just as that of a normal person and doesn't show any sign  of dyslexia. It also indicates that you have strong focus and function to execute while writing.",
    "Potential":"Hey there ! Your handwriting indicates that you have a slight chance of having dyslexia. You might be finding it problematic to apply the right amount of pressure while writing and to maintain your sentence alignment.",
    "Higher potential":"Hey there ! Your handwriting indicates that you have a high chance of having dyslexia. You might be finding it problematic to apply the right amount of pressure while writing and to maintain your sentence alignment. You might also be struggling with focus and function to execute while writing.",
    "Dyslexia":"Hey there ! Your handwriting indicates that you have a very high chance of having dyslexia. You might be finding it problematic to apply the right amount of pressure while writing"
}

def ocr1(file_name, image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/alymer/.googleCreds/credentials.json"
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.document_text_detection(image=image)

    docText = response.full_text_annotation.text
    ocr=docText.rstrip('\n')
    return(ocr)
##############################################################################
#Func word passer converts ocr output to list of words 
def wordpasser(ocr_text):
    word=ocr_text.replace(".","")
    #print(type(word))
    x=[]
    for item in word.split():
        x.append(item)
    #print(x)
    return(x)
##############################################################################   
def Marker1(word):
    #print(image_path)
    ocr_text=word
    flag1=0
    lowercase_letters = [c for c in ocr_text if c.islower()]
    uppercase_letters = [c for c in ocr_text if c.isupper()]
    #print (lowercase_letters)
    #print (uppercase_letters)
    #full_stop_counter = ocr_text.count('.')
    #print("fullstop cout:",full_stop_counter)
    #x=[sampl_no,'1',ocr_text.rstrip('\n'),1,'Dyslexia-Unwanted Capitalisation']
    #y=[sampl_no,'1',ocr_text.rstrip('\n'),0,'normal']
    if(len(uppercase_letters)>=2):
        flag1+=1
        return(flag1)
    else:
        return(0)
#############################################################################

def marker2(word):
    from spellchecker import SpellChecker
    from textblob import TextBlob 
    from textblob import Word
    r_word=Word(word)
    y=r_word.spellcheck()
    #print(y)
    flag2=0
    for i in range(len(y)):
        s=y[i][0]
        j=0
        if(flag2>=1):
            break
        else:
            for character in str(s):
                if(len(s)!=len(r_word)):
                    break
                elif(character==r_word[j]):
                    #print(r_word[j])
                    j=j+1
                    continue
                elif(character=='b' and r_word[j]=='d'):
                    flag2=flag2+1
                    #print("found")
                elif(character=='d' and r_word[j]=='b'):
                    flag2=flag2+1
                    #print("found")
                elif(character=='p' and r_word[j]=='q'):
                    flag2=flag2+1
                elif(character=='q' and r_word[j]=='p'):
                    flag2=flag2+1
                elif(character=='w' and r_word[j]=='m'):
                    flag2=flag2+1
                j=j+1
    #x=[sampl_no,'2',word,1,'Dyslexia-Mirror writing']
    #y=[sampl_no,'2',word,0,'normal']
    return(flag2)
##############################################################################

def marker4(file_name, image_path):
    text=ocr1(file_name,image_path)
    sentences=sent_tokenize(text)
    my_list1=[]
    my_list2=[]
    for sentence in sentences:
        words = spell.split_words(sentence)
        my_list1.extend(words)
        my_list2.append(words)


    #print(my_list2)
    my_list3=[]
    count =0
    for i in my_list2:
        x=spell.unknown(i)
        y=len(x)
        my_list3.append(y)
# count=0

    my_list3
    ans1=0
    x=len(my_list3)
    y=int(x)
    if (y%2==0):
        y=y/2
    else:
        y=((y+1)/2)-1

    y=int(y)

    first_half = my_list3[0:y]
    second_half = my_list3[y:]
    sum1=sum(first_half)
    sum2=sum(second_half)

    if sum2>sum1:
        return(1)#dyslexia
    else:
        return(0)
        
def marker5(word):
    ocr_text=word
    from spellchecker import SpellChecker
    from textblob import TextBlob 
    from textblob import Word
    ocr_texts=ocr_text.rstrip('\n')
    #print("ocr:",ocr_texts)
    words=ocr_texts.rstrip('\n')
    
    word=words.replace(" ","")
    #print("found:",word)
    word=Word(word)
   # spell = SpellChecker()
    y=word.spellcheck()
    #print(y)
    array=[]
    for i in range(len(y)):
        array.append(y[i][0])
        #print(array)
        if(word==y[i][0]):
            return(0)
        elif set(word)==set(y[i][0]):
            return(1)
            break
        elif(i>6):
            return(0)
        else:
            return(0)




sentence_array=[]

def marker7(text):
    my_doc = nlp(text)

# Create list of word tokens
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    from spacy.lang.en.stop_words import STOP_WORDS

# Create list of word tokens after removing stopwords
    filtered_sentence =[] 
    length=len(filtered_sentence)
    if(length==0):
        return(1)
        
    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 
    #print(token_list)
    #print(filtered_sentence)
    z=[]
    for x in filtered_sentence:
        #print(porter.stem(x))
        z.append(porter.stem(x))

    y=len(z)
    if (y%2==0):
        y=y/2
    else:
        y=(y+1)/2
    y=int(y)
    z.pop(y-1)
    z

    x=' '.join(z)
    #x

    length_string = len(x)
    first_length = round(length_string / 2)
    first_half = x[0:first_length]
    second_half = x[first_length:]

    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    doc_1=first_half
    doc_2=second_half

    documents = [doc_1, doc_2]
# Create the Document Term Matrix
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)
# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  index=['doc_1', 'doc_2'])


    from sklearn.metrics.pairwise import cosine_similarity
    ans=cosine_similarity(df, df)
    x1=ans[0][0]
    x2=ans[1][0]
    x3=ans[0][1]
    x4=ans[1][1]
    ans2=x1*x2+x3*x4
    #print(ans2)
    if(ans2>0.4):
        return(ans2)
    else:
        return(0)
##############################################################################

import numpy as np
import argparse
import cv2
import math
# construct the argument parse and parse the arguments

def marker8(file_name, image_path):
# load the image from disk
    image = cv2.imread(image_path)

# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

# otherwise, just take the inverse of the angle to make
# it positive
    else:
        angle = -angle

# rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
    #print("angle: {:.3f}".format(angle))
    #cv2.imshow("Input", image)

    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey(0)
    return(abs(angle))



import os, io
from google.cloud import vision 
import cv2
import pandas as pd

def ocrfrm1(file_name, image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/alymer/.googleCreds/credentials.json"

    client = vision.ImageAnnotatorClient()
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    # construct an iamge instance
    image = types.Image(content=content)
    
    """
    # or we can pass the image url
    image = vision.types.Image()
    image.source.image_uri = 'https://edu.pngfacts.com/uploads/1/1/3/2/11320972/grade-10-english_orig.png'
    """
    
    # annotate Image Response
    response = client.text_detection(image=image)  # returns TextAnnotation
    df = pd.DataFrame(columns=['locale', 'description'])
    
    texts = response.text_annotations
    for text in texts:
        df = df.append(
            dict(
                locale=text.locale,
                description=text.description
            ),
            ignore_index=True
        )
    y=df['description'][0]
    #print(df['description'][0])
    return y
#word passer passes ocr output word by word to check for dyslexia indicators

############################## Driver code ##########################################
def runInference(uniqueFilename):
    file_name= os.path.join("uploads/", uniqueFilename)
    sampl_no=1
    image_path = os.path.join("uploads/", uniqueFilename)
    ocr=ocr1(file_name, image_path)
    ocr_list=wordpasser(ocr)
    ocrofmarker1=ocrfrm1(file_name, image_path)
    ocr_list2=wordpasser(ocrofmarker1)
    #list of words in the ocr sentance 
    flag2=0
    flag1=0
    flag5=0
    flag7=0
    #print("ocrfm1:",ocrofmarker1)
    for words in ocr_list2:
        #print(words)
        flag1=Marker1(words)+flag1
    for word in ocr_list:
        #print(word)
        flag2=marker2(word)+flag2
        flag5=marker5(word)+flag5
    #print("Marker1count:",flag1)
    #print("Marker2count:",flag2)
    #sentence_array
    para=""
    for item in ocr.split():
        para=para+" "+item
    #print("para:",para)
    sentence_array=sent_tokenize(para)
    #sentence_array
    flag4=marker4(file_name, image_path)
    #print("Marker4 count:",flag4)
    #print("Marker5count:",flag5)
    for i in sentence_array:
        flag7=flag7+marker7(i)
    #print("Marker7count:",flag7)

    flag8=marker8(file_name, image_path)
    #print("Marker8count:",flag8)
    total=flag1+flag2+flag4+flag5

    if(flag1<1 and flag2<1 and flag4<2 and flag5<2 and (flag8>-0.2 or flag8<0.2)):
        return {"Heading":"Normal", "content":contentDictionary["Normal"]}
    elif(flag8>0.8 or flag7>2):
        return "Potential"
        return {"Heading":"Potential", "content":contentDictionary["Potential"]}
    elif((flag8>0.8 or flag4>0) or total>0):
        return "Higher Potential"
        return {"Heading":"Higher Potential", "content":contentDictionary["Higher Potential"]}
    elif((flag8>0.8 or flag4>0) or total>1):
        return "Dyslexia"
        return {"Heading":"Dyslexia", "content":contentDictionary["Dyslexia"]}
    else:
        return "Normal"
        return {"Heading":"Normal", "content":contentDictionary["Normal"]}

