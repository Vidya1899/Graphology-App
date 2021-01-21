#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import numpy as np
import sys
import cv2 as cv
import math
import pandas as pd
# from baseline2 import count_through_a_list as count_through_a_list


def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def get_f2(leng,width):
    # print(leng,width)
    width = 35
    if leng >= (0.8 * width):
        return 'high'
    elif leng >= (0.5 * width):
        return 'medium'
    else:
        return 'low'


def get_hori(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    #src = cv.imread(argv[0], cv.IMREAD_COLOR)
    # src = cv.imread(argv, cv.IMREAD_COLOR)
    src = argv

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv)
        return -1

    # Show source image
    # cv.imshow("src", src)
    # [load_image]

    # [gray]
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src

    # Show gray image
    # show_wait_destroy("gray", gray)
    # [gray]

    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    # Show binary image
    # show_wait_destroy("binary", bw)
    # [bin]

    #kernel = np.ones((3, 3), np.uint8)
    #bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    #show_wait_destroy("close", bw)

    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    # vertical = np.copy(bw)
    # [init]

    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # horizontal_size = cols
    leng = 0

    # Create structure element for extracting horizontal lines through morphology operations
    # horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    # horizontal = cv.erode(horizontal, horizontalStructure)
    # horizontal = cv.dilate(horizontal, horizontalStructure)
    # horizontal = cv.morphologyEx(horizontal, cv.MORPH_CLOSE, horizontalStructure)

    # show_wait_destroy("horizontal", horizontal)
    # print('horiz',horizontal)

    #print(horizontal.shape)
    #print(horizontal[6])
    sum_hori = 0
    max_hori = 0
    fin_row = 0

    for row in range(horizontal.shape[0]):
        sum_hori = sum(horizontal[row])
        # print(sum_hori)
        #if (sum_hori > max_now) and (sum_hori != 0):
        if (sum_hori > max_hori):
            #lines.append(horizontal[row])
            line = horizontal[row]
            max_hori = sum_hori
            fin_row = row
            #leng = sum(horizontal[row]/255)

    # print('line',line)
    # print(horizontal)
    # print(horizontalStructure)
    #print("leng:",int(sum(line)/255))
    leng = int(sum(line)/255)

    send_leng = get_f2(leng,horizontal.shape[1])

    # linesP = cv.HoughLinesP(horizontal, 1, np.pi / 180, 50, None, 50, 10)
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    #minLineLength - Minimum length of line.Line segments shorter than this are rejected.
    #maxLineGap - Maximum allowed gap between line segments to treat them as single line.
    # Python: cv2.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) ? lines
    # Parameters:image ? 8-bit, single-channel binary source image. The image may be modified by the function.
    # lines ? Output vector of lines. Each line is represented by a 4-element vector   ,
    # where   and   are the ending points of each detected line segment.
    # rho ? Distance resolution of the accumulator in pixels.
    # theta ? Angle resolution of the accumulator in radians.
    # threshold ? Accumulator threshold parameter. Only those lines are returned that get enough votes (   ).
    # minLineLength ? Minimum line length. Line segments shorter than that are rejected.
    # maxLineGap ? Maximum allowed gap between points on the same line to link them.

    linesP = cv.HoughLinesP(horizontal, 1, np.pi / 180, 50 , 30, 10)

    # print('no of lines',len(linesP))

    opt = 99


    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(src, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
            # cv.imshow('the t',src)
            # cv.waitKey(0)
            # cv.circle(cpy,(l[0],l[1]),2,(0, 255, 255),2)
            x1,y1,x2,y2 = l[0], l[1], l[2], l[3]

            cpy = src

            deltaY = y2 - y1
            deltaX = x2 - x1
            # print(x1,y1,x2,y2,deltaX,deltaY)
            angleInDegrees = math.atan2(deltaY, deltaX) * 180 / np.pi
            # print(angleInDegrees)

            # show_wait_destroy("cpy", cpy)
            print("line p")

            if abs(angleInDegrees) <= 45:
                if angleInDegrees > 0 :
                    opt = 1
                else:
                    opt = 0
                break



    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #         cv.line(src, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)


    #check row position in the image
    # print(horizontal)
    # print(horizontal.shape)
    # print(fin_row)

    height = horizontal.shape[0]

    temp = (fin_row * 100) / height
    # print(temp)

    res = ''
    if (temp >= 70):
        res = 'high'
    elif (temp >= 50):
        res = 'medium'
    else:
        res = 'low'


    # Show extracted horizontal lines
    # show_wait_destroy("horizontal", horizontal)
    # show_wait_destroy("src", src)
    # [horiz]

    # deltaY = y2 - y1
    # deltaX = x2 - x1
    #
    # angleInDegrees = math.atan2(deltaY, deltaX) * 180 / np.pi
    # print(angleInDegrees)

    """# [vert]
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)

    # Show extracted vertical lines
    show_wait_destroy("vertical", vertical)
    # [vert]

    # [smooth]
    # Inverse vertical image
    vertical = cv.bitwise_not(vertical)
    show_wait_destroy("vertical_bit", vertical)

    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''

    # Step 1
    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 3, -2)
    show_wait_destroy("edges", edges)

    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    show_wait_destroy("dilate", edges)

    # Step 3
    smooth = np.copy(vertical)

    # Step 4
    smooth = cv.blur(smooth, (2, 2))

    # Step 5
    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]

    # Show final result
    show_wait_destroy("smooth - final", vertical)
    # [smooth]"""

    #return 0


    return send_leng,res,opt

if __name__ == "__main__":
    print(get_hori(sys.argv[1]))
