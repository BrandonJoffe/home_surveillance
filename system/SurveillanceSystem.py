
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
# Code used in this project included opensource software (openface)
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

import atexit
from subprocess import Popen, PIPE
import os.path
import sys

import logging
import threading
import time

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders


import Camera
import openface
import aligndlib
import ImageProcessor

from flask import Flask, render_template, Response, redirect, url_for, request
import Camera
from flask.ext.socketio import SocketIO,send, emit #Socketio depends on gevent

#Get paths for models
#//////////////////////////////////////////////////////////////////////////////////////////////
start = time.time()
np.set_printoptions(precision=2)



#///////////////////////////////////////////////////////////////////////////////////////////////
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )
                              
class Surveillance_System(object):
    
   def __init__(self):

        self.training = True
        self.cameras = []
        self.camera_threads = []
        self.people_processing_threads = []
        self.svm = None

        self.video_frame1 = None
        self.video_frame2 = None
        self.video_frame3 = None

        fileDir = os.path.dirname(os.path.realpath(__file__))
        luaDir = os.path.join(fileDir, '..', 'batch-represent')
        modelDir = os.path.join(fileDir, '..', 'models')
        dlibModelDir = os.path.join(modelDir, 'dlib')
        openfaceModelDir = os.path.join(modelDir, 'openface')

        parser = argparse.ArgumentParser()
        parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.", default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))                  
        parser.add_argument('--networkModel', type=str, help="Path to Torch network model.", default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))                   
        parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)                    
        parser.add_argument('--cuda', action='store_true')
        parser.add_argument('--unknown', type=bool, default=False, help='Try to predict unknown people')
                            
        self.args = parser.parse_args()
        self.align = openface.AlignDlib(self.args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.args.networkModel, imgDim=self.args.imgDim,  cuda=self.args.cuda) 


        
       

        #default IP cam
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        self.cameras.append(Camera.VideoCamera("debugging/Test.mov"))
        #self.cameras.append(Camera.VideoCamera("debugging/Test.mov"))
        #self.cameras.append(Camera.VideoCamera("debugging/example_01.mp4"))

        #processing frame thread
        for i, cam in enumerate(self.cameras):
          self.proccesing_lock = threading.Lock()
          thread = threading.Thread(name='process_thread_' + str(i),target=self.process_frame,args=(cam,))
          thread.daemon = False
          self.camera_threads.append(thread)
          thread.start()

        for i, cam in enumerate(self.cameras):
          #self.proccesing_lock = threading.Lock()
          thread = threading.Thread(name='people_process_thread_' + str(i),target=self.people_processing,args=(cam,))
          thread.daemon = False
          self.people_processing_threads.append(thread)
          thread.start()


   def initialize(self):
        start = time.time()
        #align raw images  and place them in "aligned-images"
        aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",self.args.dlibFacePredictor,self.args.imgDim)
        print("\nAligning images took {} seconds.".format(time.time() - start))
        
        done = False
        start = time.time()
        #Build Classification Model from aligned images
        done = self.generate_representation(luaDir)
        if done is True:
            print("Representation Generation (Classification Model) took {} seconds.".format(time.time() - start))
            start = time.time()
            #Train Model
            self.train("generated-embeddings/","LinearSvm",-1)
            print("Training took {} seconds.".format(time.time() - start))
        else:
            print("Generate representation did not return True")

        #open cameras with Threads
        #Get Notification Details
        #Start processing

   def process_frame(self,camera):
      logging.debug('Processing Frames')

      while True:  
            frame = camera.read_frame()
            frame = ImageProcessor.resize(frame)
            height, width, channels = frame.shape

            with camera.frame_lock: #aquire lock
              camera.faceBoxes, camera.rgbFrame  = ImageProcessor.detectdlib_face(frame,height, width )
              camera.processed_frame = ImageProcessor.draw_rects_dlib(frame, camera.faceBoxes)


          
   def people_processing(self,camera):   
      logging.debug('Ready to process faces')
      detectedFaces = 0
      while True:  
          with camera.frame_lock: #aquire lock
            if camera.faceBoxes is not None:
               detectedFaces  = len(camera.faceBoxes)
               for bb in camera.faceBoxes:
              
                    alignedFace = ImageProcessor.align_face(camera.rgbFrame,bb)
                    camera.unknownPeople.append(alignedFace) # add to array so that rgbFrame can be released earlier rather than waiting for recognition
                    #cv2.imwrite("face.jpg", alignedFace)

          for face in camera.unknownPeople:
              predictions = ImageProcessor.face_recognition(camera,face)
              with camera.people_dict_lock:
                if camera.people.has_key(predictions['name']):

                    if camera.people[predictions['name']].confidence < predictions['confidence']:

                        camera.people[predictions['name']].confidence = predictions['confidence']
                        if (predictions['confidence'] > 70):
                            cv2.imwrite("notification/image.png", camera.processed_frame)
                            self.send_notification_alert(predictions['name'],camera.people[predictions['name']].confidence)

                else: 
                    ret, jpeg = cv2.imencode('.jpg', face) #convert to jpg to be viewed by client
                    face_mpeg = jpeg.tostring()
                    camera.people[predictions['name']] = Person(face_mpeg, predictions['confidence'])
          camera.unknownPeople = []
          
            

   def generate_representation(self,fileDir):
        #2 Generate Representation 
        self.cmd = ['/usr/bin/env', 'th', os.path.join(fileDir, 'main.lua'),'-outDir',  "generated-embeddings/" , '-data', "aligned-images/"]                 
        if self.args.cuda:
            self.cmd.append('-cuda')
        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)

        def exitHandler():
            if self.p.poll() is None:
                self.p.kill()
        atexit.register(exitHandler) 

        return True

   def train(self,workDir,classifier,ldaDim):
      print("Loading embeddings.")
      fname = "{}/labels.csv".format(workDir)
      labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
      labels = map(itemgetter(1),
                   map(os.path.split,
                       map(os.path.dirname, labels)))  # Get the directory.
      fname = "{}/reps.csv".format(workDir)
      embeddings = pd.read_csv(fname, header=None).as_matrix()
      le = LabelEncoder().fit(labels)
      labelsNum = le.transform(labels)
      nClasses = len(le.classes_)
      print("Training for {} classes.".format(nClasses))

      if classifier == 'LinearSvm':
          clf = SVC(C=1, kernel='linear', probability=True)
      elif classifier == 'GMM':
          clf = GMM(n_components=nClasses)

      if ldaDim > 0:
          clf_final = clf
          clf = Pipeline([('lda', LDA(n_components=ldaDim)),
                          ('clf', clf_final)])

      clf.fit(embeddings, labelsNum)

      fName = "{}/classifier.pkl".format(workDir)
      print("Saving classifier to '{}'".format(fName))
      with open(fName, 'w') as f:
          pickle.dump((le, clf), f)

   def send_notification_alert(self,name,confidence):


      # code produced in this tutorial - http://naelshiab.com/tutorial-send-email-python/

      fromaddr = "bjjoffe@gmail.com"
      toaddr = "bjjoffe@gmail.com"
       
      msg = MIMEMultipart()
       
      msg['From'] = fromaddr
      msg['To'] = toaddr
      msg['Subject'] = "HOME SURVEILLANCE NOTIFICATION"
       
      body = name + " was detected with a confidence of " + str(confidence) + "\n\n"
       
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
      server.login(fromaddr, "Jofhouse021")
      text = msg.as_string()
      server.sendmail(fromaddr, toaddr, text)
      server.quit()
      

   def add_face():
      return

   def add_notifications():
      return



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
class Person(object):
    person_count = 0

    def __init__(self,thumbnail, confidence):       
        #self.personCoord = personCoord
        self.identity = "unknown_" + str(Person.person_count)
        self.confidence = confidence
        self.thumbnail = thumbnail
        Person.person_count += 1 
        #self.tracker = dlib.correlation_tracker()
    
    def get_identity(self):
        return self.identity

    def set_identity(self, id):
        self.identity = id

    # def recognize_face(self):     
    #    return

    # def update_position(self, newCoord):
    #    self.personCoord = newCoord

    # def get_current_position(self):
    #    return self.personCoord

    # def start_tracking(self,img):
    #  self.tracker.start_track(img, dlib.rectangle(self.personCoord))
    
    # def update_tracker(self,img):    
    #    self.tracker.update(img)

    # def get_position(self):
    #    return self.tracker.get_position()

    # def find_face(self):     
    #    return

   # tracking = FaceTracking(detect_min_size=detect_min_size,
   #                          detect_every=detect_every,
   #                          track_min_overlap_ratio=track_min_overlap_ratio,
   #                          track_min_confidence=track_min_confidence,
   #                          track_max_gap=track_max_gap)

        




