# ImageProcessor.
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

import cv2
import numpy as np
import os
import glob
#from skimage import io
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
#import imutils
import threading

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import time
start = time.time()
from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import Camera
import openface

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

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)
neuralNet_lock = threading.Lock()

# net = openface.TorchNeuralNet(../models/openface/nn4.small2.v1.t7, imgDim=args.imgDim,
#                               cuda=args.cuda)



facecascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
uppercascade = cv2.CascadeClassifier("cascades/haarcascade_upperbody.xml")
eyecascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")


def detect_faces(camera,img,width,height):

    buf = np.asarray(img)
    rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]

    annotatedFrame = np.copy(buf)
    #start = time.time()
    bbs = align.getAllFaceBoundingBoxes(rgbFrame)
    #print("Face detection took {} seconds.".format(time.time() - start))

    for bb in bbs:

        bl = (bb.left(), bb.bottom()) # (x, y)
        tr = (bb.right(), bb.top()) # (x+w,y+h)

        print("\n=====================================================================")
        print("Face Being Processed")
        start = time.time()
        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)                                                     
        if alignedFace is None:
            cv2.rectangle(annotatedFrame, bl, tr, color=(255, 150, 150), thickness=2)
            print("//////////////////////  FACE COULD NOT BE ALIGNED  //////////////////////////")
            continue
        print("Face Alignment took {} seconds.".format(time.time() - start))

        #///////////////////////////////////////////////////////////////////

        #//////////////////////////////////////////////////////////////////
        cv2.imwrite("facedebug.png", alignedFace)
        with neuralNet_lock:
            persondict = recognize_face("generated-embeddings/classifier.pkl",alignedFace,net)
        if persondict is None:
            print("//////////////////////  FACE COULD NOT BE RECOGNIZED  //////////////////////////")
            continue
        else:
            print("=====================================================================")
            if persondict['confidence'] > 0.60:
                cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 200), thickness=2)
            else:  
                cv2.rectangle(annotatedFrame, bl, tr, color=(100, 255, 255),thickness=2) 
            cv2.putText(annotatedFrame,  str(persondict['name']) + " " + str(math.ceil(persondict['confidence']*100))+ "%", (bb.left()-15, bb.bottom() + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                    color=(152, 255, 204), thickness=1)

    return annotatedFrame

def motion_detector(camera,frame):
        #calculate mean standard deviation then determine if motion has actually accurred
        text = "Unoccupied"
        occupied = False
        # resize the frame, convert it to grayscale, filter and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        gray = cv2.medianBlur(gray,9) 
        #fgbg = cv2.BackgroundSubtractorMOG2(history=5, varThreshold=16, bShadowDetection = False)
        #thresh = fgbg.apply(frame ,learningRate=1.0/5)
        cv2.imwrite("grayfiltered.jpg", gray)

        #initialise and build some history
        if camera.history == 0:
            camera.current_frame = gray
            camera.history +=1
            return occupied 
        elif camera.history == 1:
            camera.previous_frame = camera.current_frame
            camera.current_frame = gray
            #camera.next_frame = gray
            camera.meanframe = cv2.addWeighted(camera.previous_frame,0.5,camera.current_frame,0.5,0)
            cv2.imwrite("avegrayfiltered.jpg", camera.meanframe)
            camera.history +=1
            return occupied 
        elif camera.history == 20:
            camera.previous_frame = camera.current_frame
            camera.current_frame = gray
            #camera.next_frame = gray
            camera.history = 0

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(camera.meanframe , gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite("motion.jpg", thresh)
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image

        thresh = cv2.dilate(thresh, None, iterations=2)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 8000 or cv2.contourArea(c) > 80000:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # text = "Occupied"
            occupied = True
        # draw the text and timestamp on the frame
        # cv2.putText(frame, "Room Status: {}".format(text), (10, 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        #     (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # if len(cnts) > 0:
        #     return True
        # return False

        camera.history +=1

        return occupied
    
def resize(frame):
    r = 420.0 / frame.shape[1]
    dim = (420, int(frame.shape[0] * r))
    # perform the actual resizing of the image and show it
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)    
    return frame  

def crop(image, box):
    return image[box.top():box.bottom(), box.left():box.right()]

    #img[y: y + h, x: x + w] 
    #bl = (bb.left(), bb.bottom()) # (x1, y1)
    #tr = (bb.right(), bb.top()) # (x+w,y+h) = (x2,y2)
    #(x1, y1), (x2, y2)

        # x1 y1 -------------
        # -------------------
        # -------------------
        # -------------------
        # -------------------
        # ----------------x2 y2

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

def draw_person(image, bl, tr):
   cv2.rectangle(image, bl, tr, color=(100, 100, 255),thickness=2) 

def draw_text(image, persondict):
    cv2.putText(image,  str(persondict['name']) + " " + str(math.ceil(persondict['confidence']*100))+ "%", (bb.left()-15, bb.bottom() + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25,
                    color=(152, 255, 204), thickness=1)


def pre_processing(image):
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     gray = cv2.equalizeHist(gray)
     return gray

def draw_rects_cv(img, rects, color=(0, 40, 255)):

    overlay = img.copy()
    output = img.copy()
    
    count = 1
    for x, y, w, h in rects:
      
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

        #(x1, y1), (x2, y2)

        # x1 y1 -------------
        # -------------------
        # -------------------
        # -------------------
        # -------------------
        # ----------------x2 y2
    return output

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


def detect_cascadeface(img, cascade):

    
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20), flags = cv2.CASCADE_SCALE_IMAGE)
   

    return rects

def detect_cascade(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60), flags = cv2.CASCADE_SCALE_IMAGE)
    return rects

def detect_people_hog(image):
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


def detect_people_cascade(image):
    #image = pre_processing(image)
    rects = detect_cascade(image, uppercascade)
    
    image = draw_rects_cv(image, rects,color=(0, 255, 0))  
    return image

def detectopencv_face(image):
    frame = image.copy()
    image = pre_processing(image)

    #start = time.time()
    rects = detect_cascadeface(image, facecascade)
    #Ttime = time.time() - start

    #frame = draw_rects_cv(frame, rects)  

    #lineString = "speed: " + str(Ttime )
    #writeToFile("detections.txt",lineString)
   
    return rects

def detectdlib_face(img,height,width):

    buf = np.asarray(img)
    rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]

    annotatedFrame = np.copy(buf)

    # start = time.time()
    bbs = align.getAllFaceBoundingBoxes(rgbFrame)
    # Ttime = time.time() - start
    #print("Face detection took {} seconds.".format(time.time() - start))
    # lineString = "speed: " + str(Ttime )
    # writeToFile("detections.txt",lineString)

    return bbs, annotatedFrame

def convertImageToNumpyArray(img,height,width): #numpy array used by dlib for image operations
    buf = np.asarray(img)
    rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]

    annotatedFrame = np.copy(buf)
    return annotatedFrame


def align_face(rgbFrame,bb):

    landmarks = align.findLandmarks(rgbFrame, bb)
    alignedFace = align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)                                                     
    if alignedFace is None:  
        print("//////////////////////  FACE COULD NOT BE ALIGNED  //////////////////////////")
        return alignedFace

    print("//////////////////////  FACE ALIGNED  ////////////////////// ")
    return alignedFace

def face_recognition(camera,alignedFace):

    with neuralNet_lock:
        persondict = recognize_face("generated-embeddings/classifier.pkl",alignedFace, net)

    if persondict is None:
        print("//////////////////////  FACE COULD NOT BE RECOGNIZED  //////////////////////////")
        return persondict
    else:
        print("//////////////////////  FACE RECOGNIZED  ////////////////////// ")
        return persondict

def recognize_face(classifierModel,img,net):

    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)

    if getRep(img,net) is None:
        return None
    rep = getRep(img,net).reshape(1, -1)
    start = time.time()
    predictions = clf.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = int(math.ceil(predictions[maxI]*100))

    print("Recognition took {} seconds.".format(time.time() - start))
    print("Recognized {} with {:.2f} confidence.".format(person, confidence))
    #if isinstance(clf, GMM):
    #    dist = np.linalg.norm(rep - clf.means_[maxI])
    #    print("  + Distance from the mean: {}".format(dist))

    persondict = {'name': person, 'confidence': confidence}
    return persondict


def getRep(alignedFace,net):

    bgrImg = alignedFace
    if bgrImg is None:
        print("unable to load image")
        return None

    alignedFace = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    start = time.time()
    rep = net.forward(alignedFace)
    #print("Neural network forward pass took {} seconds.".format(  time.time() - start))
    return rep


def writeToFile(filename,lineString): #Used for writing testing data to file

       f = open(filename,"a") 
       f.write(lineString + "\n")    
       f.close()
