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



#facecascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
#uppercascade = cv2.CascadeClassifier("cascades/haarcascade_upperbody.xml")
#eyecascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

def recognize_face(classifierModel,img):

    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)

    if getRep(img) is None:
        return None
    rep = getRep(img).reshape(1, -1)
    start = time.time()
    predictions = clf.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]

    print("Recognition took {} seconds.".format(time.time() - start))
    print("Recognized {} with {:.2f} confidence.".format(person, confidence))
    #if isinstance(clf, GMM):
    #    dist = np.linalg.norm(rep - clf.means_[maxI])
    #    print("  + Distance from the mean: {}".format(dist))

    persondict = {'name': person, 'confidence': confidence}
    return persondict


def getRep(alignedFace):

    bgrImg = alignedFace
    if bgrImg is None:
        print("unable to load image")
        return None

    alignedFace = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    start = time.time()
    rep = net.forward(alignedFace)
    #print("Neural network forward pass took {} seconds.".format(  time.time() - start))
    return rep


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

        bl = (bb.left(), bb.bottom())
        tr = (bb.right(), bb.top())
                        
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

        cv2.imwrite("facedebug.png", alignedFace)

        persondict = recognize_face("generated-embeddings/classifier.pkl",alignedFace)
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
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25,
                    color=(152, 255, 204), thickness=1)

    return annotatedFrame
        

def crop(image, box):
    return image[box.bottom():box.bottom()+box.top(), box.left():box.left()+box.right()]

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

def pre_processing(image):
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     gray = cv2.equalizeHist(gray)
     return gray

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def detect_cascade(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=2, minNeighbors=4, minSize=(20, 20), flags = cv2.CASCADE_SCALE_IMAGE)
    return rects

def detect_people_hog(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(image,winStride=(7,7),padding=(16,16), scale=1.05)

    filtered_detections = []
  
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            filtered_detections.append(r)
            draw_person(image, r)  
 
    return image


def detect_people_cascade(image):
    image = pre_processing(image)
    rects = detect_cascade(image, uppercascade)
    
    for person in rects:
       draw_person(image, person) 
    return image

def detectprocess_face(image):
    frame = image.copy()
    image = pre_processing(image)
    rects = detect_cascade(image, facecascade)
  
    for person in rects:
        x, y, w, h = person
        faceimg = crop (frame, x, y, w, h)
        draw_person(frame, person) 
        alignedfaceimg = face_functions.process_face(faceimg,w,h)
        
    
    return frame




