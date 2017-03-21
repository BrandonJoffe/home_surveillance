
# Surveillance System Controller.
# Brandon Joffe
# 2016
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
# Code used in this project included opensource software (Openface)
# developed by Brandon Amos
# Copyright 2015-2016 Carnegie Mellon University


import time
import argparse
import cv2
import os
import pickle
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.mixture import GMM
import dlib
import atexit
from subprocess import Popen, PIPE
import os.path
import sys
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
from datetime import datetime, timedelta
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders
import requests
import json
from openface.data import iterImgs
import Camera
import FaceRecogniser
import openface
import aligndlib
import ImageUtils
import random
import psutil
import math

# Get paths for models
# //////////////////////////////////////////////////////////////////////////////////////////////

fileDir = os.path.dirname(os.path.realpath(__file__))
luaDir = os.path.join(fileDir, '..', 'batch-represent')
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', 
                    type=str, help="Path to dlib's face predictor.", 
                    default=os.path.join(dlibModelDir , "shape_predictor_68_face_landmarks.dat"))                  
parser.add_argument('--networkModel', 
                   type=str, help="Path to Torch network model.", 
                   default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))                   
parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)                    
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False, help='Try to predict unknown people')                  
args = parser.parse_args()

start = time.time()
np.set_printoptions(precision=2)

try:
    os.makedirs('logs', exist_ok=True)  # Python>3.2
except TypeError:
    try:
        os.makedirs('logs')
    except OSError as exc:  # Python >2.5
        print "logging directory already exist"

logger = logging.getLogger()
formatter = logging.Formatter("(%(threadName)-10s) %(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = RotatingFileHandler("logs/surveillance.log", maxBytes=10000000, backupCount=10)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

#logging.basicConfig(level=logging.DEBUG,
#                    format='(%(threadName)-10s) %(message)s',
#                    )
                  
class SurveillanceSystem(object):
   """ The SurveillanceSystem object is the heart of this application.
   It provides all the central proccessing and ties everything
   together. It generates camera frame proccessing threads as 
   well as an alert monitoring thread. A camera frame proccessing 
   thread can process a camera using 5 different processing methods.
   These methods aim to allow the user to adapt the system to their 
   needs and can be found in the process_frame() function. The alert 
   monitoring thread continually checks the system state and takes 
   action if a particular event occurs. """ 

   def __init__(self):

        self.recogniser = FaceRecogniser.FaceRecogniser()
        self.trainingEvent = threading.Event() # Used to holt processing while training the classifier 
        self.trainingEvent.set() 
        self.drawing = True 
        self.alarmState = 'Disarmed' # Alarm states - Disarmed, Armed, Triggered
        self.alarmTriggerd = False
        self.alerts = [] # Holds all system alerts
        self.cameras = [] # Holds all system cameras
        self.camerasLock  = threading.Lock() # Used to block concurrent access of cameras []
        self.cameraProcessingThreads = []
        self.peopleDB = []
        self.confidenceThreshold = 20 # Used as a threshold to classify a person as unknown

        # Initialization of alert processing thread 
        self.alertsLock = threading.Lock()
        self.alertThread = threading.Thread(name='alerts_process_thread_',target=self.alert_engine,args=())
        self.alertThread.daemon = False
        self.alertThread.start()

        # Used for testing purposes
        ###################################
        self.testingResultsLock = threading.Lock()
        self.detetectionsCount = 0
        self.trueDetections = 0
        self.counter = 0
        ####################################

        self.get_face_database_names() # Gets people in database for web client

        #//////////////////////////////////////////////////// Camera Examples ////////////////////////////////////////////////////
        #self.cameras.append(Camera.IPCamera("testing/iphoneVideos/singleTest.m4v","detect_recognise_track",False)) # Video Example - uncomment and run code
        # self.cameras.append(Camera.IPCamera("http://192.168.1.33/video.mjpg","detect_recognise_track",False))
        
        # processing frame threads 
        for i, cam in enumerate(self.cameras):       
          thread = threading.Thread(name='frame_process_thread_' + str(i),target=self.process_frame,args=(cam,))
          thread.daemon = False
          self.cameraProcessingThreads.append(thread)
          thread.start()

   def add_camera(self, camera):
        """Adds new camera to the System and generates a 
        frame processing thread"""
        self.cameras.append(camera)
        thread = threading.Thread(name='frame_process_thread_' + 
                                 str(len(self.cameras)),
                                 target=self.process_frame,
                                 args=(self.cameras[-1],))
        thread.daemon = False
        self.cameraProcessingThreads.append(thread)
        thread.start()

   def remove_camera(self, camID):
        """remove a camera to the System and kill its processing thread"""
        self.cameras.pop(camID)
        self.cameraProcessingThreads.pop(camID)
        self.captureThread.stop = False

   def process_frame(self,camera):
        """This function performs all the frame proccessing.
        It reads frames captured by the IPCamera instance,
        resizes them, and performs 1 of 5 functions"""
        logger.debug('Processing Frames')
        state = 1
        frame_count = 0;  
        FPScount = 0 # Used to calculate frame rate at which frames are being processed
        FPSstart = time.time()
        start = time.time()
        stop = camera.captureThread.stop
        
        while not stop:

             frame_count +=1
             logger.debug("Reading Frame")
             frame = camera.read_frame()
             if frame == None or np.array_equal(frame, camera.tempFrame):  # Checks to see if the new frame is the same as the previous frame
                 continue
             frame = ImageUtils.resize(frame)
             height, width, channels = frame.shape

            # Frame rate calculation 
             if FPScount == 6:
                 camera.processingFPS = 6/(time.time() - FPSstart)
                 FPSstart = time.time()
                 FPScount = 0

             FPScount += 1
             camera.tempFrame = frame
        
             ##################################################################################################################################################
             #<###########################################################> MOTION DETECTION <################################################################>
             ##################################################################################################################################################

             if camera.cameraFunction == "detect_motion":
                 camera.motion, mframe = camera.motionDetector.detect_movement(frame, get_rects = False) 
                 camera.processing_frame = mframe
                 if camera.motion == False:
                    logger.debug('//// NO MOTION DETECTED /////')
                    continue
                 else:
                    logger.debug('/// MOTION DETECTED ///')



             ##################################################################################################################################################
             #<#####################################################> FACE DETECTION AND RECOGNTIION <#########################################################>
             ##################################################################################################################################################

             elif camera.cameraFunction == "detect_recognise":
                    # This approach peroforms basic face detection and 
                    # recognition using OpenCV, Dlib and Openface

                    training_blocker = self.trainingEvent.wait()  

                    frame = cv2.flip(frame, 1) # converts frame from BGR (OpenCV format) to RGB (Dlib format)
                    camera.faceBoxes = camera.faceDetector.detect_faces(frame,camera.dlibDetection) 
                    if self.drawing == True:
                         frame = ImageUtils.draw_boxes(frame, camera.faceBoxes, camera.dlibDetection)
                    camera.processing_frame = frame
         
                    logger.info('////  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' //')
                    for face_bb in camera.faceBoxes: 
                        
                        # Used to reduce false positives from opencv haar cascade detector.
                        # If face isn't detected using more rigorous paramters in the detectMultiscale() function read the next frame               
                        if camera.dlibDetection == False:
                              x, y, w, h = face_bb
                              face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                              faceimg = ImageUtils.crop(frame, face_bb, dlibRect = True)
                              if len(camera.faceDetector.detect_cascadeface_accurate(faceimg)) == 0:
                                    continue

                        # returns a dictionary that contains name, confidence and representation and an alignedFace (numpy array)
                        predictions, alignedFace = self.recogniser.make_prediction(frame,face_bb) 

                        with camera.peopleDictLock:
                          # If the person has already been detected and his new confidence is greater update persons details, otherwise create a new person
                          if camera.people.has_key(predictions['name']): 
                              if camera.people[predictions['name']].confidence < predictions['confidence']:
                                  camera.people[predictions['name']].confidence = predictions['confidence']

                                  if camera.people[predictions['name']].confidence > self.confidenceThreshold:
                                     camera.people[predictions['name']].identity = predictions['name']

                                  camera.people[predictions['name']].set_thumbnail(alignedFace) 
                                  camera.people[predictions['name']].add_to_thumbnails(alignedFace)  
                                  camera.people[predictions['name']].set_time()
                          else: 
                              if predictions['confidence'] > self.confidenceThreshold:
                                  camera.people[predictions['name']] = Person(predictions['rep'],predictions['confidence'], alignedFace, predictions['name'])
                              else: 
                                  camera.people[predictions['name']] = Person(predictions['rep'],predictions['confidence'], alignedFace, "unknown")
                   
                    camera.processing_frame = frame # Used for streaming proccesed frames to client and email alerts, but mainly used for testing purposes

              ##################################################################################################################################################
              #<#####################################> MOTION DETECTION EVENT FOLLOWED BY FACE DETECTION AND RECOGNITION <#####################################>
              ##################################################################################################################################################

             elif camera.cameraFunction == "motion_detect_recognise":
                # When motion is detected, consecutive frames are proccessed for faces.
                # If no faces are detected for longer than 30 seconds the thread goes back to
                # looking for motion 

                 training_blocker = self.trainingEvent.wait()  

                 if state == 1: # If no faces have been found or there has been no movement

                     camera.motion, mframe = camera.motionDetector.detect_movement(frame, get_rects = False)   
          
                     if camera.motion == True:
                        logger.debug('////////////////////// MOTION DETECTED //////////////////////')
                        state = 2
                        camera.processing_frame = mframe
                     else:
                        logger.debug('////////////////////// NO MOTION DETECTED //////////////////////')
                     continue

                 elif state == 2: # If motion has been detected
                    if frame_count == 0:
                        start = time.time()
                        frame_count += 1

                    frame = cv2.flip(frame, 1)
                    camera.faceBoxes = camera.faceDetector.detect_faces(frame,camera.dlibDetection)
                    if self.drawing == True:
                        frame = ImageUtils.draw_boxes(frame, camera.faceBoxes, camera.dlibDetection)
                 
                    camera.processing_frame = frame

                    if len(camera.faceBoxes) == 0:
                        if (time.time() - start) > 30.0:
                            logger.info('//  No faces found for ' + str(time.time() - start) + ' seconds - Going back to Motion Detection Mode')
                            state = 1
                            frame_count = 0;
                    else:
                        logger.info('////  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' ////')
                        # frame = cv2.flip(frame, 1)
                        for face_bb in camera.faceBoxes: 
                      
                            if camera.dlibDetection == False:
                                  x, y, w, h = face_bb
                                  face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                                  faceimg = ImageUtils.crop(frame, face_bb, dlibRect = True)
                                  if len(camera.faceDetector.detect_cascadeface_accurate(faceimg)) == 0:
                                        continue

                            predictions, alignedFace = self.recogniser.make_prediction(frame,face_bb)

                            with camera.peopleDictLock:
                              if camera.people.has_key(predictions['name']):
                                  if camera.people[predictions['name']].confidence < predictions['confidence']:
                                      camera.people[predictions['name']].confidence = predictions['confidence']

                                      if camera.people[predictions['name']].confidence > self.confidenceThreshold:
                                         camera.people[predictions['name']].identity = predictions['name']

                                      camera.people[predictions['name']].set_thumbnail(alignedFace)  
                                      camera.people[predictions['name']].add_to_thumbnails(alignedFace) 
                                      camera.people[predictions['name']].set_time()
                              else: 
                                  if predictions['confidence'] > self.confidenceThreshold:
                                      camera.people[predictions['name']] = Person(predictions['rep'],predictions['confidence'], alignedFace, predictions['name'])
                                  else: 
                                      camera.people[predictions['name']] = Person(predictions['rep'],predictions['confidence'], alignedFace, "unknown")

                        start = time.time() # Used to go back to motion detection state of 30s of not finding a face
                        camera.processing_frame = frame

              ###################################################################################################################################################################
              #<#####################################>  MOTION DETECTION OBJECT SEGMENTAION FOLLOWED BY FACE DETECTION AND RECOGNITION <#####################################>
              ####################################################################################################################################################################
      
             elif camera.cameraFunction == "segment_detect_recognise":
                    # This approach uses background subtraction to segement a region of
                    # interest that is likely to contain a person. The region is cropped from
                    # the frame and face detection is performed on a much smaller image. This 
                    # improves proccessing performance but is highly dependent upon the accuracy of 
                    # the background model generated by the MotionDetector object.

                    training_blocker = self.trainingEvent.wait()       
                    camera.motion, peopleRects  = camera.motionDetector.detect_movement(frame, get_rects = True)   

                    if camera.motion == False:
                       camera.processing_frame = frame
                       logger.debug('////-- NO MOTION DETECTED --////')
                       continue

                    logger.debug('///// MOTION DETECTED /////')
                    if self.drawing == True:
                        frame = ImageUtils.draw_boxes(frame, peopleRects, False)

                    for x, y, w, h in peopleRects:
                      
                        logger.debug('//// Proccessing People Segmented Areas ///')
                        bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h)) 
                        personimg = ImageUtils.crop(frame, bb, dlibRect = True)
                       
                        personimg = cv2.flip(personimg, 1)
                        camera.faceBoxes = camera.faceDetector.detect_faces(personimg,camera.dlibDetection)
                        if self.drawing == True:
                            camera.processing_frame = ImageUtils.draw_boxes(frame, peopleRects, False)

                        for face_bb in camera.faceBoxes: 

                              if camera.dlibDetection == False:
                                    x, y, w, h = face_bb
                                    face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                                    faceimg = ImageUtils.crop(personimg, face_bb, dlibRect = True)
                                    if len(camera.faceDetector.detect_cascadeface_accurate(faceimg)) == 0:
                                          continue
                              logger.info('/// Proccessing Detected faces ///')

                              predictions, alignedFace = self.recogniser.make_prediction(personimg,face_bb)

                              with camera.peopleDictLock:
                                if camera.people.has_key(predictions['name']):
                                    if camera.people[predictions['name']].confidence < predictions['confidence']:
                                        camera.people[predictions['name']].confidence = predictions['confidence']
                                        camera.people[predictions['name']].set_thumbnail(alignedFace)  
                                        camera.people[predictions['name']].add_to_thumbnails(alignedFace) 
                                        camera.people[predictions['name']].set_time()
                                else: 
                                    if predictions['confidence'] > self.confidenceThreshold:
                                        camera.people[predictions['name']] = Person(predictions['rep'],predictions['confidence'], alignedFace, predictions['name'])
                                    else: 
                                        camera.people[predictions['name']] = Person(predictions['rep'],predictions['confidence'], alignedFace, "unknown")
              
              ############################################################################################################################################################################
              #<#####################################>  MOTION DETECTION OBJECT SEGMENTAION FOLLOWED BY FACE DETECTION, RECOGNITION AND TRACKING <#####################################>
              #############################################################################################################################################################################

             elif camera.cameraFunction == "detect_recognise_track":
                # This approach incorporates background subtraction to perform person tracking 
                # and is the most efficient out of the all proccesing funcions above. When
                # a face is detected in a region a Tracker object it generated, and is updated
                # every frame by comparing the last known region of the person, to new regions
                # produced by the motionDetector object. Every update of the tracker a detected 
                # face is compared to the person's face of whom is being tracked to ensure the tracker
                # is still tracking the correct person. This is acheived by comparing the prediction
                # and the the l2 distance between their embeddings (128 measurements that represent the face).
                # If a tracker does not overlap with any of the regions produced by the motionDetector object
                # for some time the Tracker is deleted. 

                training_blocker = self.trainingEvent.wait()  # Wait if classifier is being trained 

                logger.debug('//// detect_recognise_track 1 ////')
                peopleFound = False
                camera.motion, peopleRects  = camera.motionDetector.detect_movement(frame, get_rects = True)   
                logger.debug('//// detect_recognise_track  2 /////')
          
                if camera.motion == False:
                   camera.processing_frame = frame
                   logger.debug('///// NO MOTION DETECTED /////')
                   continue

                if self.drawing == True:
                    camera.processing_frame = ImageUtils.draw_boxes(frame, peopleRects, False)

                logger.debug('//// MOTION DETECTED //////')

               
                for x, y, w, h in peopleRects:

                    peopleFound = True
                    person_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h)) 
                    personimg = ImageUtils.crop(frame, person_bb, dlibRect = True)   # Crop regions of interest 

                    personimg = cv2.flip(personimg, 1) 

                    tracked = False
                    # Iterate through each tracker and compare there current psotiion
                    for i in xrange(len(camera.trackers) - 1, -1, -1): 
                        
                        if camera.trackers[i].overlap(person_bb):
                           logger.debug("=> Updating Tracker <=")
                           camera.trackers[i].update_tracker(person_bb)
                           # personimg = cv2.flip(personimg, 1)
                           camera.faceBoxes = camera.faceDetector.detect_faces(personimg,camera.dlibDetection)  
                           logger.debug('//////  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' /////')
                           if len(camera.faceBoxes) > 0:
                               logger.info("Found " + str(len(camera.faceBoxes)) + " faces.")
                           for face_bb in camera.faceBoxes: 

                                if camera.dlibDetection == False:
                                    x, y, w, h = face_bb
                                    face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                                    faceimg = ImageUtils.crop(personimg, face_bb, dlibRect = True)
                                    if len(camera.faceDetector.detect_cascadeface_accurate(faceimg)) == 0:
                                          continue

                                predictions, alignedFace =  self.recogniser.make_prediction(personimg,face_bb)
                        
                                if predictions['confidence'] > self.confidenceThreshold:
                                    predictedName = predictions['name']
                                else:
                                    predictedName = "unknown"
                                # If only one face is detected
                                if len(camera.faceBoxes) == 1:
                                    # if not the same person check to see if tracked person is unknown and update or change tracker accordingly
                                    # l2Distance is between 0-4 Openface found that 0.99 was the average cutoff between the same and different faces
                                    # the same face having a distance less than 0.99 
                                    if self.recogniser.getSquaredl2Distance(camera.trackers[i].person.rep ,predictions['rep']) > 0.99 and (camera.trackers[i].person.identity != predictedName): 
                                      
                                            alreadyBeenDetected = False
                                            with camera.peopleDictLock:
                                                    for ID, person in camera.people.iteritems():  # iterate through all detected people in camera
                                                        # if the person has already been detected continue to track that person - use same person ID
                                                        if person.identity == predictedName or self.recogniser.getSquaredl2Distance(person.rep ,predictions['rep']) < 0.8:
                                                              
                                                                person = Person(predictions['rep'],predictions['confidence'], alignedFace, predictedName)
                                                                logger.info( "====> New Tracker for " +person.identity + " <===")
                                                                # Remove current tracker and create new one with the ID of the original person
                                                                del camera.trackers[i]
                                                                camera.trackers.append(Tracker(frame, person_bb, person,ID))
                                                                alreadyBeenDetected = True
                                                                break

                                            if not alreadyBeenDetected:
                                                    num = random.randrange(1, 1000, 1)    
                                                    strID = "person" +  datetime.now().strftime("%Y%m%d%H%M%S") + str(num) # Create a new person ID
                                                    # Is the new person detected with a low confidence? If yes, classify them as unknown
                                                    if predictions['confidence'] > self.confidenceThreshold:
                                                          person = Person(predictions['rep'],predictions['confidence'], alignedFace, predictions['name'])
                                                    else:   
                                                          person = Person(predictions['rep'],predictions['confidence'], alignedFace, "unknown")
                                                    #add person to detected people      
                                                    with camera.peopleDictLock:
                                                          camera.people[strID] = person
                                                          logger.info( "=====> New Tracker for new person <====")
                                                    del camera.trackers[i]
                                                    camera.trackers.append(Tracker(frame, person_bb, person,strID))
                                    # if it is the same person update confidence if it is higher and change prediction from unknown to identified person
                                    # if the new detected face has a lower confidence and can be classified as unknown, when the person being tracked isn't unknown - change tracker
                                    else:
                                        logger.info( "====> update person name and confidence <==")
                                        if camera.trackers[i].person.confidence < predictions['confidence']:
                                            camera.trackers[i].person.confidence = predictions['confidence']
                                            if camera.trackers[i].person.confidence > self.confidenceThreshold:
                                                camera.trackers[i].person.identity = predictions['name']
      
                                  
                                # If more than one face is detected in the region compare faces to the people being tracked and update tracker accordingly
                                else:
                                    logger.info( "==> More Than One Face Detected <==")
                                    # if tracker is already tracking the identified face make an update 
                                    if self.recogniser.getSquaredl2Distance(camera.trackers[i].person.rep ,predictions['rep']) < 0.99 and camera.trackers[i].person.identity == predictions['name']: 
                                        if camera.trackers[i].person.confidence < predictions['confidence']:
                                            camera.trackers[i].person.confidence = predictions['confidence']
                                            if camera.trackers[i].person.confidence > self.confidenceThreshold:
                                                camera.trackers[i].person.identity = predictions['name']
                                    else:
                                        # if tracker isn't tracking this face check the next tracker
                                        break
                                
                                camera.trackers[i].person.set_thumbnail(alignedFace)  
                                camera.trackers[i].person.add_to_thumbnails(alignedFace)
                                camera.trackers[i].person.set_rep(predictions['rep'])
                                camera.trackers[i].person.set_time()
                                camera.trackers[i].reset_face_pinger()
                                with camera.peopleDictLock:
                                        camera.people[camera.trackers[i].id] = camera.trackers[i].person
                           camera.trackers[i].reset_pinger()
                           tracked = True
                           break

                    # If the region is not being tracked
                    if not tracked:

                        # Look for faces in the cropped image of the region
                        camera.faceBoxes = camera.faceDetector.detect_faces(personimg,camera.dlibDetection)
                       
                        for face_bb in camera.faceBoxes:

                            if camera.dlibDetection == False:
                                  x, y, w, h = face_bb
                                  face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                                  faceimg = ImageUtils.crop(personimg, face_bb, dlibRect = True)
                                  if len(camera.faceDetector.detect_cascadeface_accurate(faceimg)) == 0:
                                        continue

                            predictions, alignedFace =  self.recogniser.make_prediction(personimg,face_bb)
                
                            alreadyBeenDetected = False
                            with camera.peopleDictLock:
                                    for ID, person in camera.people.iteritems():  # iterate through all detected people in camera, to see if the person has already been detected
                                        if person.identity == predictions['name'] or self.recogniser.getSquaredl2Distance(person.rep ,predictions['rep']) < 0.8: 
                                                if predictions['confidence'] > self.confidenceThreshold and person.confidence > self.confidenceThreshold:
                                                      person = Person(predictions['rep'],predictions['confidence'], alignedFace, predictions['name'])
                                                else:   
                                                      person = Person(predictions['rep'],predictions['confidence'], alignedFace, "unknown")
                                                logger.info( "==> New Tracker for " + person.identity + " <====")
                                   
                                                camera.trackers.append(Tracker(frame, person_bb, person,ID))
                                                alreadyBeenDetected = True
                                                break
                                         
                            if not alreadyBeenDetected:
                                    num = random.randrange(1, 1000, 1)    # Create new person ID if they have not been detected
                                    strID = "person" +  datetime.now().strftime("%Y%m%d%H%M%S") + str(num)
                                    if predictions['confidence'] > self.confidenceThreshold:
                                          person = Person(predictions['rep'],predictions['confidence'], alignedFace, predictions['name'])
                                    else:   
                                          person = Person(predictions['rep'],predictions['confidence'], alignedFace, "unknown")
                                    #add person to detected people      
                                    with camera.peopleDictLock:
                                          camera.people[strID] = person
                                    logger.info( "====> New Tracker for new person <=")
                                    camera.trackers.append(Tracker(frame, person_bb, person,strID))


                for i in xrange(len(camera.trackers) - 1, -1, -1): # starts with the most recently initiated tracker
                    if self.drawing == True:
                          bl = (camera.trackers[i].bb.left(), camera.trackers[i].bb.bottom()) # (x, y)
                          tr = (camera.trackers[i].bb.right(), camera.trackers[i].bb.top()) # (x+w,y+h)
                          cv2.rectangle(frame, bl, tr, color=(0, 255, 255), thickness=2)
                          cv2.putText(frame,  camera.trackers[i].person.identity + " " + str(camera.trackers[i].person.confidence)+ "%", (camera.trackers[i].bb.left(), camera.trackers[i].bb.top() - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                                    color=(0, 255, 255), thickness=1)
                    camera.processing_frame = frame
                    # Used to check if tracker hasn't been updated
                    camera.trackers[i].ping()
                    camera.trackers[i].faceping()

                    # If the tracker hasn't been updated for more than 10 pings delete it
                    if camera.trackers[i].pings > 10: 
                        del camera.trackers[i]
                        continue

             elif camera.cameraFunction == "testing":
             # Used for testing puposes
                    training_blocker = self.trainingEvent.wait()  
                    # tempframe = frame
                    frame = cv2.flip(frame, 1)

                    camera.faceBoxes = camera.faceDetector.detect_faces(frame,camera.dlibDetection)
                 

                    if self.drawing == True:
                         frame = ImageUtils.draw_boxes(frame, camera.faceBoxes, camera.dlibDetection)

                    camera.processing_frame = frame
                   
         
                    logger.debug('////  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' //')

                    for face_bb in camera.faceBoxes: 
                        result = ""
                        # used to reduce false positives from opencv haar cascade detector
                        if camera.dlibDetection == False:
                              x, y, w, h = face_bb
                              face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                              faceimg = ImageUtils.crop(frame, face_bb, dlibRect = True)
                              if len(camera.faceDetector.detect_cascadeface_accurate(faceimg)) == 0:
                                    continue
                        with self.testingResultsLock:
                            self.detetectionsCount += 1 
                           
                            predictions, alignedFace = self.recogniser.make_prediction(frame,face_bb)
                            cv2.imwrite('testing/results/unconstrained/alignedDetections/60/'+ str( self.detetectionsCount) +'.png',alignedFace)
                            if predictions['name'] == 'brandon-joffe':
                                  self.trueDetections += 1
                                  self.confidence_sum += predictions['confidence']
                        
                            result = str( self.detetectionsCount) + ', ' + predictions['name'] + ', ' + str(predictions['confidence'])+ ', ' + str(self.trueDetections) + ', ' + str(self.confidence_sum)
                            ImageUtils.writeToFile('testing/results/unconstrained/accuracy/results60.txt',result)

             elif camera.cameraFunction == "face_capture":
             # This will be used to capture faces for training in the surveillance environment 
             # not fully implmented - was mainly used for face capture during testing
                    training_blocker = self.trainingEvent.wait()  
                    # tempframe = frame
                    frame = cv2.flip(frame, 1)

                    camera.faceBoxes = camera.faceDetector.detect_faces(frame,camera.dlibDetection)
  
                    logger.debug('//  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' ///')


                    for face_bb in camera.faceBoxes: 
                        result = ""
                        # used to reduce false positives from opencv haar cascade detector
                        if camera.dlibDetection == False:
                              x, y, w, h = face_bb
                              face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                              faceimg = ImageUtils.crop(frame, face_bb, dlibRect = True)
                              if len(camera.faceDetector.detect_cascadeface_accurate(faceimg)) == 0:
                                    continue
                        with self.testingResultsLock:
                            self.detetectionsCount += 1 
                           
                            predictions, alignedFace = self.recogniser.make_prediction(frame,face_bb)
                            # cv2.imwrite('testing/alignedFacesForTraining/surelda/surelda'+ str(self.detetectionsCount) +'.png',alignedFace)
                            cv2.imwrite('testing/alignedFacesForTesting/tracy/tracy-'+ str(self.detetectionsCount) +'.png',alignedFace)

                    if self.drawing == True:
                       frame = ImageUtils.draw_boxes(frame, camera.faceBoxes, camera.dlibDetection)
                    camera.processing_frame = frame
                                   
   def alert_engine(self):  
        """check alarm state -> check camera -> check event -> 
        either look for motion or look for detected faces -> take action"""

        logger.debug('Alert engine starting')
        while True:
           with self.alertsLock:
              for alert in self.alerts:
                logger.debug('checking alert')
                if alert.action_taken == False: # If action hasn't been taken for event 
                    if alert.alarmState != 'All':  # Check states
                        if  alert.alarmState == self.alarmState: 
                            logger.debug('checking alarm state')
                            alert.event_occurred = self.check_camera_events(alert)
                        else:
                          continue # Alarm not in correct state check next alert
                    else:
                        alert.event_occurred = self.check_camera_events(alert)
                else:
                    if (time.time() - alert.eventTime) > 300: # Reinitialize event 5 min after event accured
                        logger.info( "reinitiallising alert: " + alert.id)
                        alert.reinitialise()
                    continue 

           time.sleep(2) # Put this thread to sleep - let websocket update alerts if need be (i.e delete or add)
  
   def check_camera_events(self,alert):   
        """Used to check state of cameras
        to determine whether an event has occurred"""

        if alert.camera != 'All':  # Check cameras   
            logger.info( "alertTest" + alert.camera)
            if alert.event == 'Recognition': #Check events
                logger.info(  "checkingalertconf "+ str(alert.confidence) + " : " + alert.person)
                for person in self.cameras[int(alert.camera)].people.values():
                    logger.info( "checkingalertconf "+ str(alert.confidence )+ " : " + alert.person + " : " + person.identity)
                    if alert.person == person.identity: # Has person been detected
                       
                        if alert.person == "unknown" and (100 - person.confidence) >= alert.confidence:
                            logger.info( "alertTest2" + alert.camera)
                            cv2.imwrite("notification/image.png", self.cameras[int(alert.camera)].processing_frame)#
                            self.take_action(alert)
                            return True
                        elif person.confidence >= alert.confidence:
                            logger.info( "alertTest3" + alert.camera)
                            cv2.imwrite("notification/image.png", self.cameras[int(alert.camera)].processing_frame)#
                            self.take_action(alert)
                            return True     
                return False # Person has not been detected check next alert       

            else:
                logger.info( "alertTest4" + alert.camera)
                if self.cameras[int(alert.camera)].motion == True: # Has motion been detected
                       logger.info( "alertTest5" + alert.camera)
                       cv2.imwrite("notification/image.png", self.cameras[int(alert.camera)].processing_frame)#
                       self.take_action(alert)
                       return True
                else:
                  return False # Motion was not detected check next alert
        else:
            if alert.event == 'Recognition': # Check events
                with self.camerasLock :
                    cameras = self.cameras
                for camera in cameras: # Look through all cameras
                    for person in camera.people.values():
                        if alert.person == person.identity: # Has person been detected
                            if alert.person == "unknown" and (100 - person.confidence) >= alert.confidence:
                                cv2.imwrite("notification/image.png", camera.processing_frame)#
                                self.take_action(alert)
                                return True
                            elif person.confidence >= alert.confidence:
                                cv2.imwrite("notification/image.png", camera.processing_frame)#
                                self.take_action(alert)
                                return True
               
                return False # Person has not been detected check next alert   

            else:
                with  self.camerasLock :
                    for camera in self.cameras: # Look through all cameras
                        if camera.motion == True: # Has motion been detected
                            cv2.imwrite("notification/image.png", camera.processing_frame)#
                            self.take_action(alert)
                            return True

                return False # Motion was not detected check next camera

   def take_action(self,alert): 
        """Sends email alert and/or triggers the alarm"""

        logger.info( "Taking action: ==" + alert.actions)
        if alert.action_taken == False: # Only take action if alert hasn't accured - Alerts reinitialise every 5 min for now
            alert.eventTime = time.time()  
            if alert.actions['email_alert'] == 'true':
                logger.info( "email notification being sent")
                self.send_email_notification_alert(alert)
            if alert.actions['trigger_alarm'] == 'true':
                logger.info( "triggering alarm1")
                self.trigger_alarm()
                logger.info( "alarm1 triggered")
            alert.action_taken = True

   def send_email_notification_alert(self,alert):
      """ Code produced in this tutorial - http://naelshiab.com/tutorial-send-email-python/"""

      fromaddr = "home.face.surveillance@gmail.com"
      toaddr = alert.emailAddress

      msg = MIMEMultipart()
       
      msg['From'] = fromaddr
      msg['To'] = toaddr
      msg['Subject'] = "HOME SURVEILLANCE"
       
      body = "NOTIFICATION ALERT:" +  alert.alertString + ""
       
      msg.attach(MIMEText(body, 'plain'))
       
      filename = "image.png"
      attachment = open("notification/image.png", "rb")    
      part = MIMEBase('application', 'octet-stream')
      part.set_payload((attachment).read())
      encoders.encode_base64(part)
      part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
       
      msg.attach(part)
       
      server = smtplib.SMTP('smtp.gmail.com', 587)
      server.starttls()
      server.login(fromaddr, "facialrecognition")
      text = msg.as_string()
      server.sendmail(fromaddr, toaddr, text)
      server.quit()

   def add_face(self,name,image, upload):
      """Adds face to directory used for training the classifier"""

      if upload == False:
          path = fileDir + "/aligned-images/" 
      else:
          path = fileDir + "/training-images/"         
      num = 0
    
      if not os.path.exists(path + name):
        try:
          logger.info( "Creating New Face Dircectory: " + name)
          os.makedirs(path+name)
        except OSError:
          logger.info( OSError)
          return False
          pass
      else:
         num = len([nam for nam in os.listdir(path +name) if os.path.isfile(os.path.join(path+name, nam))])

      logger.info( "Writing Image To Directory: " + name)
      cv2.imwrite(path+name+"/"+ name + "_"+str(num) + ".png", image)
      self.get_face_database_names()

      return True


   def get_face_database_names(self):
      """Gets all the names that were most recently 
      used to train the classifier"""

      path = fileDir + "/aligned-images/" 
      self.peopleDB = []
      for name in os.listdir(path):
        if (name == 'cache.t7' or name == '.DS_Store' or name[0:7] == 'unknown'):
          continue
        self.peopleDB.append(name)
        logger.info("Known faces in our db for: " + name + " ")
      self.peopleDB.append('unknown')

   def change_alarm_state(self):
      """Sends Raspberry PI a resquest to change the alarm state.
      192.168.1.35 is the RPI's static IP address port 5000 is used 
      to access the flask application."""

      r = requests.post('http://192.168.1.35:5000/change_state', data={"password": "admin"})
      alarm_states = json.loads(r.text)

      logger.info(alarm_states)
      if alarm_states['state'] == 1:
          self.alarmState = 'Armed' 
      else:
          self.alarmState = 'Disarmed'       
      self.alarmTriggerd = alarm_states['triggered']

   def trigger_alarm(self):
       """Sends Raspberry PI a resquest to change to trigger the alarm.
      192.168.1.35 is the RPI's static IP address port 5000 is used 
      to access the flask application."""

       r = requests.post('http://192.168.1.35:5000/trigger', data={"password": "admin"})
       alarm_states = json.loads(r.text) 
    
       logger.info(alarm_states)

       if alarm_states['state'] == 1:
           self.alarmState = 'Armed' 
       else:
           self.alarmState = 'Disarmed' 
       
       self.alarmTriggerd = alarm_states['triggered']
       logger.info(self.alarmTriggerd )

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
class Person(object):
    """Person object simply holds all the
    person's information for other processes
    """
    person_count = 0

    def __init__(self,rep,confidence = 0, face = None, name = "unknown"):  

        if "unknown" not in name: # Used to include unknown-N from Database
            self.identity = name
        else:
            self.identity = "unknown"
       
        self.count = Person.person_count
        self.confidence = confidence  
        self.thumbnails = []
        self.face = face
        self.rep = rep # Face representation
        if face is not None:
            ret, jpeg = cv2.imencode('.jpg', face) # Convert to jpg to be viewed by client
            self.thumbnail = jpeg.tostring()
        self.thumbnails.append(self.thumbnail) 
        Person.person_count += 1 
        now = datetime.now() + timedelta(hours=2)
        self.time = now.strftime("%A %d %B %Y %I:%M:%S%p")
        self.istracked = False
   
    def set_rep(self, rep):
        self.rep = rep

    def set_identity(self, identity):
        self.identity = identity

    def set_time(self): # Update time when person was detected
        now = datetime.now() + timedelta(hours=2)
        self.time = now.strftime("%A %d %B %Y %I:%M:%S%p")

    def set_thumbnail(self, face):
        self.face = face
        ret, jpeg = cv2.imencode('.jpg', face) # Convert to jpg to be viewed by client
        self.thumbnail = jpeg.tostring()

    def add_to_thumbnails(self, face):
        ret, jpeg = cv2.imencode('.jpg', face) # Convert to jpg to be viewed by client
        self.thumbnails.append(jpeg.tostring())

class Tracker:
    """Keeps track of person position"""

    tracker_count = 0

    def __init__(self, img, bb, person, id):
        self.id = id 
        self.person = person
        self.bb = bb
        self.pings = 0
        self.facepings = 0

    def reset_pinger(self):
        self.pings = 0

    def reset_face_pinger(self):
        self.facepings = 0

    def update_tracker(self,bb):
        self.bb  = bb 
        
    def overlap(self, bb):
        p = float(self.bb.intersect(bb).area()) / float(self.bb.area())
        return p > 0.2

    def ping(self):
        self.pings += 1

    def faceping(self):
        self.facepings += 1

class Alert(object): 
    """Holds all the alert details and is continually checked by 
    the alert monitoring thread"""

    alert_count = 1

    def __init__(self,alarmState,camera, event, person, actions, emailAddress, confidence):   
        logger.info( "alert_"+str(Alert.alert_count)+ " created")
       

        if  event == 'Motion':
            self.alertString = "Motion detected in camera " + camera 
        else:
            self.alertString = person + " was recognised in camera " + camera + " with a confidence greater than " + str(confidence)

        self.id = "alert_" + str(Alert.alert_count)
        self.event_occurred = False
        self.action_taken = False
        self.camera = camera
        self.alarmState = alarmState
        self.event = event
        self.person = person
        self.confidence = confidence
        self.actions = actions
        if emailAddress == None:
            self.emailAddress = "bjjoffe@gmail.com"
        else:
            self.emailAddress = emailAddress

        self.eventTime = 0

        Alert.alert_count += 1

    def reinitialise(self):
        self.event_occurred = False
        self.action_taken = False

    def set_custom_alertmessage(self,message):
        self.alertString = message



