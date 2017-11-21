import numpy as np
from numpy import *
import os
import time
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib
import cv2
import matplotlib.pyplot as plt

def Tbar(int, void): pass

def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 1, 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 1, )
        print ((x + ix)/2, (y + iy)/2)


drawing = False
playing = False
ix, iy = -1, -1
#cv2.namedWindow('image')
#cv2.setMouseCallback('image', draw_rect)
frame = np.zeros((10, 10, 3), np.uint8)
gb_size = 4
kernel_size = 2
mb_size = 22
erode_times = 1
dilate_times = 4
threshold_diff = 100
number_frame = 0
pre_gray = 0
ret = True
contours_number = 5
kernel = np.ones((kernel_size * 2 + 1, kernel_size * 2 + 1), np.uint8)
cv2.namedWindow("settings", 1)
cv2.createTrackbar("gb_size", "settings", gb_size, 20, Tbar)
cv2.createTrackbar("kernel_size", "settings", kernel_size, 20, Tbar)
cv2.createTrackbar("threshold_diff", "settings", threshold_diff, 255, Tbar)
cv2.createTrackbar("erode_times1", "settings", erode_times, 20, Tbar)
cv2.createTrackbar("dilate_times", "settings", dilate_times, 20, Tbar)
cv2.createTrackbar("contours_number", "settings", contours_number, 5, Tbar)
cv2.createTrackbar("mb_size", "settings", mb_size, 100, Tbar)

cap = cv2.VideoCapture(r'C:\Users\cr\Desktop\dataset\VOC2007\test16.MOV')
# cap = cv2.VideoCapture(0)

while (cap.isOpened()):

    if (ret == True):
        #settings
        gb_size = cv2.getTrackbarPos("gb_size", "settings")
        kernel_size = cv2.getTrackbarPos("kernel_size", "settings")
        threshold_diff = cv2.getTrackbarPos("threshold_diff", "settings")
        erode_times = cv2.getTrackbarPos("erode_times1", "settings")
        dilate_times = cv2.getTrackbarPos("dilate_times", "settings")
        contours_number = cv2.getTrackbarPos("contours_number", "settings")
        mb_size = cv2.getTrackbarPos("mb_size", "settings")
        #waitkey'q'&'p'
        k = cv2.waitKey(25) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('p'):
            playing = not playing
            print (playing)
        #cv2.imshow('image', frame)

        if playing == True:
            number_frame += 1
            #print('frame:%d' %number_frame)

            if (number_frame == 1 ):
                ret, frame = cap.read()
                blur = cv2.GaussianBlur(frame, (gb_size * 2 + 1, gb_size * 2 + 1), 0)
                gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                pre_gray = gray.copy()
                #cv2.imshow('video_src', frame)
                continue
            else:  # number_frame<545 and number_frame>1 or number_frame==545
                ret, frame = cap.read()
                blur = cv2.GaussianBlur(frame, (gb_size * 2 + 1, gb_size * 2 + 1), 0)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #cv2.imshow('video_src', frame)

                gray_diff = gray.copy()
                cv2.subtract(pre_gray, gray, gray_diff)
                cv2.imshow('video_diff', gray_diff)

                ret0, binary = cv2.threshold(abs(gray_diff), threshold_diff, 255, cv2.THRESH_BINARY)
                cv2.imshow("binary_src", binary)

                x = 0
                while (x < erode_times):
                    binary = cv2.erode(binary, kernel, 1)
                    x += 1
                x = 0
                while (x < dilate_times):
                    binary = cv2.dilate(binary, kernel, 1)
                    x += 1
                # binary = cv2.medianBlur(binary, mb_size * 2 + 1)
                cv2.imshow('binary_ED', binary)

                binary3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                # gray_diff_2 = cv2.cvtColor(abs(gray_diff), cv2.COLOR_GRAY2BGR)
                result = cv2.bitwise_and(binary3, frame)
                # cv2.imshow('result', result)


                pre_gray = gray.copy()

                _, contours, hierarchy = cv2.findContours(abs(gray_diff), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) <= contours_number:
                    #cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
                    #print ("found %d " % (len(contours)))
                    for i in range(len(contours)):
                        x, y, w, h = cv2.boundingRect(contours[i])
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        print ((2*x + w) / 2, (2*y + h) / 2)

                cv2.imshow("result", frame)


            if (number_frame > 544):
                ret = False

    else:
        break
cap.release()
cv2.destroyAllWindows()

