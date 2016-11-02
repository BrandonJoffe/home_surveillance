# ImageUtils.
# Brandon Joffe
# 2016
#
# Copyright 2016, Brandon Joffe, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code used in this project included opensource software (openface)
# developed by Brandon Amos
# Copyright 2015-2016 Carnegie Mellon University

# This is a utilities script used for testing, resizing images etc

import cv2
import numpy as np
import os
import glob
import dlib
import sys
import argparse
import imagehash
import json
from PIL import Image
import urllib
import base64
import pickle
import math
import datetime
import threading
import logging
import csv
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import time
start = time.time()
from operator import itemgetter
import numpy as np
import pandas as pd
import Camera
import openface

np.set_printoptions(precision=2)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
args = parser.parse_args()


cascade_lock = threading.Lock()
facecascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
uppercascade = cv2.CascadeClassifier("cascades/haarcascade_upperbody.xml")
eyecascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
detector = dlib.get_frontal_face_detector()
    
def resize(frame):
    r = 640.0 / frame.shape[1]
    dim = (640, int(frame.shape[0] * r))
    # Resize frame to be processed
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)    
    return frame 

def resize_mjpeg(frame):
    r = 320.0 / frame.shape[1]
    dim = (320, 200)#int(frame.shape[0] * r))
    # perform the actual resizing of the image and show it
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)    
    return frame  

def crop(image, box, dlibRect = False):

    if dlibRect == False:
       x, y, w, h = box
       return image[y: y + h, x: x + w] 

    return image[box.top():box.bottom(), box.left():box.right()]

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

def draw_boxes(image, rects, dlibrects):
   if dlibrects:
       image = draw_rects_dlib(image, rects)
   else:
       image = draw_rects_cv(image, rects)
   return image


def draw_rects_cv(img, rects, color=(0, 40, 255)):
    overlay = img.copy()
    output = img.copy()
    count = 1
    for x, y, w, h in rects:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output
   
def draw_rects_dlib(img, rects, color = (0, 255, 255)):
    overlay = img.copy()
    output = img.copy()
      
    for bb in rects:
        bl = (bb.left(), bb.bottom()) # (x, y)
        tr = (bb.right(), bb.top()) # (x+w,y+h)
        cv2.rectangle(overlay, bl, tr, color, thickness=2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)       
    return output

def draw_text(image, persondict):
    cv2.putText(image,  str(persondict['name']) + " " + str(math.ceil(persondict['confidence']*100))+ "%", (bb.left()-15, bb.bottom() + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25,
                    color=(152, 255, 204), thickness=1)

def draw_rect(img,x,y,w,h, color=(0, 40, 255)):
    overlay = img.copy()
    output = img.copy()   
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output

def draw_rects_dlib(img, rects):
    overlay = img.copy()
    output = img.copy()
    for bb in rects:
        bl = (bb.left(), bb.bottom()) # (x, y)
        tr = (bb.right(), bb.top()) # (x+w,y+h)
        cv2.rectangle(overlay, bl, tr, color=(0, 255, 255), thickness=2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)       
    return output

def detect_upper_cascade(img):
    rects = uppercascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    return rects

def detect_people_hog(image):
    image = rgb_pre_processing(image)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(image,winStride=(30,30),padding=(16,16), scale=1.1)
    filtered_detections = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            filtered_detections.append(r)
    image = draw_rects_cv(image, filtered_detections)  
 
    return image

def pre_processing(image):  
     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
     cl1 = clahe.apply(grey)
     cv2.imwrite('clahe_2.jpg',cl1)
     return cl1

def detect_people_cascade(image):
    image = rgb_pre_processing(image)
    rects = detect_cascade(image, uppercascade)
    image = draw_rects_cv(image, rects,color=(0, 255, 0))  
    return image

def detectopencv_face(image):
    image = pre_processing(image)
    rects = detect_cascadeface(image)
  
    return rects

def detectlight_face(image):
    image = pre_processing(image)
    rectscv = detect_cascade(image,facecascade)
    processedimg = draw_rects_cv(image, rectscv)
    rectsdlib = detectdlibgrey_face(image)
    processedimg = draw_rects_dlib(processedimg, rectsdlib)

    cv2.imwrite('RGBlighting_Normalization.jpg',processedimg)


def detectdlibgrey_face(grey):
    bbs = detector(grey,1)
    return bbs

def detectdlib_face(img,height,width):
    buf = np.asarray(img)
    rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]

    annotatedFrame = np.copy(buf)
    bbs = align.getAllFaceBoundingBoxes(rgbFrame)

    return bbs #, annotatedFrame


def convertImageToNumpyArray(img,height,width): # Numpy array used by dlib for image operations
    buf = np.asarray(img)
    rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]
    annotatedFrame = np.copy(buf)
    return annotatedFrame


def writeToFile(filename,lineString): # Used for writing testing data to file
       f = open(filename,"a") 
       f.write(lineString + "\n")    
       f.close()


