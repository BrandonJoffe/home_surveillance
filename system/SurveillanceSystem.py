
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

import requests
import json

import Camera
import openface
import aligndlib
import ImageProcessor

from flask import Flask, render_template, Response, redirect, url_for, request
import Camera
from flask.ext.socketio import SocketIO,send, emit #Socketio depends on gevent

from instapush import Instapush, App #Used for push notifications
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
        self.trainingEvent = threading.Event()
        self.trainingEvent.set()

        self.alarmState = 'Disarmed' #disarmed, armed, triggered
        self.alarmTriggerd = False
        self.alerts = []
        self.cameras = []

        self.camera_threads = []
        self.camera_facedetection_threads = []
        self.people_processing_threads = []
        self.svm = None

        self.video_frame1 = None
        self.video_frame2 = None
        self.video_frame3 = None

        self.fileDir = os.path.dirname(os.path.realpath(__file__))
        self.luaDir = os.path.join(self.fileDir, '..', 'batch-represent')
        self.modelDir = os.path.join(self.fileDir, '..', 'models')
        self.dlibModelDir = os.path.join(self.modelDir, 'dlib')
        self.openfaceModelDir = os.path.join(self.modelDir, 'openface')

        parser = argparse.ArgumentParser()
        parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.", default=os.path.join(self.dlibModelDir, "shape_predictor_68_face_landmarks.dat"))                  
        parser.add_argument('--networkModel', type=str, help="Path to Torch network model.", default=os.path.join(self.openfaceModelDir, 'nn4.small2.v1.t7'))                   
        parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)                    
        parser.add_argument('--cuda', action='store_true')
        parser.add_argument('--unknown', type=bool, default=False, help='Try to predict unknown people')
                            
        self.args = parser.parse_args()
        self.align = openface.AlignDlib(self.args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.args.networkModel, imgDim=self.args.imgDim,  cuda=self.args.cuda) 

        #self.change_alarmState()
        #self.trigger_alarm()
        
        #self.trainClassifier()  # add faces to DB and train classifier

        #default IP cam
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("debugging/iphone_distance1080pHD.m4v"))
        self.cameras.append(Camera.VideoCamera("debugging/Test.mov"))
        #self.cameras.append(Camera.VideoCamera("debugging/rotationD.m4v"))
        #self.cameras.append(Camera.VideoCamera("debugging/example_01.mp4"))
        
        #processing frame threads- for detecting motion and face detection


       
        for i, cam in enumerate(self.cameras):
          self.proccesing_lock = threading.Lock()
          thread = threading.Thread(name='frame_process_thread_' + str(i),target=self.process_frame,args=(cam,))
          thread.daemon = False
          self.camera_threads.append(thread)
          thread.start()

        #Threads for alignment and recognition

        for i, cam in enumerate(self.cameras):
          #self.proccesing_lock = threading.Lock()
          thread = threading.Thread(name='face_process_thread_' + str(i),target=self.people_processing,args=(cam,))
          thread.daemon = False
          self.people_processing_threads.append(thread)
          thread.start()

        #Thread for alert processing  
        self.alerts_lock = threading.Lock()
        thread = threading.Thread(name='alerts_process_thread_',target=self.alert_engine,args=())
        thread.daemon = False
        thread.start()


        #open cameras with Threads
        #Get Notification Details
        #Start processing

   def process_frame(self,camera):
        logging.debug('Processing Frames')
        state = 1
        frame_count = 0;
        start = time.time()


        #img = cv2.imread('debugging/lighting.jpg')

        #cv2.imwrite("frames/lighting" +str(frame_count) +".jpg", img)
        #bl = (bb.left(), bb.bottom()) # (x1, y1)
        #tr = (bb.right(), bb.top()) # (x+w,y+h) = (x2,y2)
        
        while True:  

             frame = camera.read_frame()
             frame = ImageProcessor.resize(frame)
             #height, width, channels = frame.shape

           
             camera.temp_frame = frame
             # with camera.frame_lock: #aquire lock
             #            camera.faceBoxes, camera.rgbFrame  = ImageProcessor.detectdlib_face(frame,height, width)
             
             # if state == 1:
             #     camera.motion = ImageProcessor.motion_detector(camera,frame)
             #     if camera.motion == True:
             #        logging.debug('Motion Detected - Looking for faces in Face Detection Mode')
             #        state = 2
             #     camera.processed_frame = frame

             # elif state == 2:

             #    if frame_count == 0:
             #      start = time.time()
             #      frame_count += 1
             #    with camera.frame_lock: #aquire lock
             #            camera.faceBoxes, camera.rgbFrame  = ImageProcessor.detectdlib_face(frame,height, width)
             #    if len(camera.faceBoxes) == 0:
             #      if (time.time() - start) > 10.0:
             #        logging.debug('No faces found for ' + str(time.time() - start) + ' seconds - Going back to Motion Detection Mode')
             #        state = 1
             #        frame_count = 0;
             #        camera.processed_frame = frame
             #    else:
             #        start = time.time()
             #        camera.processed_frame = ImageProcessor.draw_rects_dlib(frame, camera.faceBoxes)

             # camera.faceBoxes, camera.rgbFrame  = ImageProcessor.detectdlib_face(frame, height, width )
      

       #frame  = ImageProcessor.detect_people_cascade(frame)
       #camera.processed_frame = frame #ImageProcessor.draw_rects_dlib(frame, camera.faceBoxes)
                      
          
   def people_processing(self,camera):   
      logging.debug('Ready to process faces')
      detectedFaces = 0
      faceBoxes = None
      while True:  

          blocker = self.trainingEvent.wait()

          if camera.temp_frame is None:
              continue

          camera.processed_frame = camera.temp_frame
          height, width, channels = camera.processed_frame.shape

          with camera.frame_lock: #aquire lock
              camera.faceBoxes, camera.rgbFrame  = ImageProcessor.detectdlib_face(camera.processed_frame ,height, width)

          with camera.frame_lock: #aquire lock   
              faceBoxes = camera.faceBoxes

          if camera.faceBoxes is not None:
              detectedFaces  = len(faceBoxes)
              for bb in faceBoxes:
                  alignedFace = ImageProcessor.align_face(camera.rgbFrame,bb)
                  camera.unknownPeople.append(alignedFace) # add to array so that rgbFrame can be released earlier rather than waiting for recognition
                  #cv2.imwrite("face.jpg", alignedFace)
          for face in camera.unknownPeople:
              predictions = ImageProcessor.face_recognition(camera,face)
              with camera.people_dict_lock:
                if camera.people.has_key(predictions['name']):

                    if camera.people[predictions['name']].confidence < predictions['confidence']:

                        camera.people[predictions['name']].confidence = predictions['confidence']
                        camera.people[predictions['name']].set_thumbnail(face)
                    
                        if (predictions['confidence'] > 70):
                            cv2.imwrite("notification/image.png", camera.processed_frame)#
                else: 
                    camera.people[predictions['name']] = Person(predictions['confidence'], face)
          camera.unknownPeople = []
          
            
   def alert_engine(self):     #check alarm state -> check camera -> check event -> either look for motion or look for detected faces -> take action
        logging.debug('Alert engine starting')
        while True:

           with self.alerts_lock:
              for alert in self.alerts:
                logging.debug('checking alert')
                if alert.action_taken == False: #if action hasn't been taken for event 
                    if alert.alarmState != 'All':  #check states
                        if  alert.alarmState == self.alarmState: 
                            logging.debug('checking alarm state')
                            alert.event_occurred = self.check_camera_events(alert)
                        else:
                          continue # alarm not in correct state check next alert
                    else:
                        alert.event_occurred = self.check_camera_events(alert)
                else:
                    if (time.time() - alert.eventTime) > 3600: # reinitialize event 1 hour after event accured
                        print "reinitiallising alert: " + alert.id
                        alert.reinitialise()
                    continue 

           time.sleep(2) #put this thread to sleep - let websocket update alerts if need be (delete or add)

  
   def check_camera_events(self,alert):   

        if alert.camera != 'All':  #check cameras               
            if alert.event == 'Recognition': #Check events

                if (self.cameras[int(alert.camera)].people[alert.person] != None): # has person been detected

                      self.take_action(alert)
                      return True

                else:
                      return False # person has not been detected check next alert
                 
            else:

                if self.cameras[int(alert.camera)].motion == True: # has motion been detected

                       self.take_action(alert)
                       return True

                else:
                  return False # motion was not detected check next alert

        else:
            if alert.event == 'Recognition': #Check events

                for camera in self.cameras: # look through all cameras

                    if (camera.people[alert.person] != None): # has person been detected

                        self.take_action(alert)
                        return True

                    else:
                        return False # person has not been detected check next camera
                 
            else:
                for camera in self.cameras: # look through all cameras

                    if camera.motion == True: # has motion been detected

                        self.take_action(alert)
                        return True

                    else:
                        return False # motion was not detected check next camera


   def take_action(self,alert): 
        print "Taking action: ======================================================="
        print alert.actions
        print "======================================================================"
        if alert.action_taken == False: #only take action if alert hasn't accured 
            alert.eventTime = time.time()
            if alert.actions['push_alert'] == 'true':
                print "\npush notification being sent\n"
                self.send_push_notification(alert.alertString)
            if alert.actions['email_alert'] == 'true':
                print "\nemail notification being sent\n"
                self.send_email_notification_alert(alert.alertString)
            if alert.actions['trigger_alarm'] == 'true':
                print "\ntriggering alarm\n"
                #trigger_alarm
            if alert.actions['notify_police'] == 'true':
                print "\nnotifying police\n"
                #notify police
            alert.action_taken = True

   def trainClassifier(self):

        self.trainingEvent.clear() #event used to hault face_processing threads to ensure no threads try access .pkl file while it is being updated

        
        path = self.fileDir + "/aligned-images/cache.t7" 
        os.remove(path) # remove cache from aligned images folder

        start = time.time()
        aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",self.args.dlibFacePredictor,self.args.imgDim)
        print("\nAligning images took {} seconds.".format(time.time() - start))
          
        done = False
        start = time.time()

        done = self.generate_representation()
           
        if done is True:
            print("Representation Generation (Classification Model) took {} seconds.".format(time.time() - start))
            start = time.time()
            #Train Model
            self.train("generated-embeddings/","LinearSvm",-1)
            print("Training took {} seconds.".format(time.time() - start))
        else:
            print("Generate representation did not return True")

        self.trainingEvent.set() #threads can continue processing

        return True
      
   def generate_representation(self):
        #2 Generate Representation 
        print "\n" + self.luaDir + "\n"
        self.cmd = ['/usr/bin/env', 'th', os.path.join(self.luaDir, 'main.lua'),'-outDir',  "generated-embeddings/" , '-data', "aligned-images/"]                 
        if self.args.cuda:
            self.cmd.append('-cuda')
        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)
        result = self.p.wait()  # wait for subprocess to finish writing to files - labels.csv reps.csv

        def exitHandler():
            if self.p.poll() is None:
                print "======================Something went Wrong============================"
                self.p.kill()
                return False
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

   def send_email_notification_alert(self,alertMessage):
      # code produced in this tutorial - http://naelshiab.com/tutorial-send-email-python/
      fromaddr = "home.face.surveillance@gmail.com"
      toaddr = "bjjoffe@gmail.com"
       
      msg = MIMEMultipart()
       
      msg['From'] = fromaddr
      msg['To'] = toaddr
      msg['Subject'] = "HOME SURVEILLANCE NOTIFICATION"
       
      body = "NOTIFICATION ALERT\n================\n" +  alertMessage + "\n"
       
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

   def send_push_notification (self,alarmMesssage): # pip install instapush 

      #insta = Instapush(user_token='57c5f710a4c48a6d45ee19ce')

      #insta.list_app()             #List all apps

      #insta.add_app(title='Home Surveillance') #Create a app named title

      app = App(appid='57c5f92aa4c48adc4dee19ce', secret='2ed5c7b8941214510a94cfe4789ddb9f')

      #app.list_event()             #List all event

      #app.add_event(event_name='FaceDetected', trackers=['message'],
      #              message='{message} face detected.')

      app.notify(event_name='FaceDetected', trackers={'message': "NOTIFICATION ALERT\n================\n" +  alarmMesssage})

   def add_face(self,name,image):

      path = self.fileDir + "/aligned-images/" 
      num = 0
    
      if not os.path.exists(path + name):
        try:
          print "Creating New Face Dircectory: " + name
          os.makedirs(path+name)
        except OSError:
          print OSError
          return False
          pass
      else:
         num = len([nam for nam in os.listdir(path +name) if os.path.isfile(os.path.join(path+name, nam))])

      print "Writing Image To Directory: " + name
      cv2.imwrite(path+name+"/"+ name + "-"+str(num) + ".png", image)

      return True

  

   def getFaceDatabaseNames(self):
      return

   def change_alarmState(self):
      r = requests.post('http://192.168.1.35:5000/change_state', data={"password": "admin"})
      alarm_states = json.loads(r.text) 
    
      print alarm_states

      if alarm_states['state'] == 1:
          self.alarmState = 'Armed' 
      else:
         self.alarmState = 'Disarmed' 
       
      self.alarmTriggerd = alarm_states['triggered']



   def trigger_alarm(self):

      r = requests.post('http://192.168.1.35:5000/trigger', data={"password": "admin"})
      alarm_states = json.loads(r.text) 
    
      print alarm_states

      if alarm_states['state'] == 1:
          self.alarmState = 'Armed' 
      else:
         self.alarmState = 'Disarmed' 
       
      self.alarmTriggerd = alarm_states['triggered']
      print self.alarmTriggerd 





#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
class Person(object):
    person_count = 0

    def __init__(self,confidence, face):       
        #self.personCoord = personCoord

        self.identity = "unknown_" + str(Person.person_count)
        self.confidence = confidence
        self.face = face
        ret, jpeg = cv2.imencode('.jpg', face) #convert to jpg to be viewed by client
        self.thumbnail = jpeg.tostring()

        Person.person_count += 1 
        #self.tracker = dlib.correlation_tracker()
    
    def get_identity(self):
        return self.identity

    def set_thumbnail(self, face):
        self.face = face
        ret, jpeg = cv2.imencode('.jpg', face) #convert to jpg to be viewed by client
        self.thumbnail = jpeg.tostring()

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

class Alert(object): #array of alerts   alert(camera,alarmstate(use switch statements), event(motion recognition),)

    alert_count = 0

    def __init__(self,alarmState,camera, event, person, actions):   
        print "\n\nalert_"+str(Alert.alert_count)+ " created\n\n"
       

        if  event == 'Motion':
            self.alertString = "Motion detected in camera " + camera 
        else:
            self.alertString = person + " was recognised in camera " + camera 

        self.id = "alert_"+str(Alert.alert_count)
        self.event_occurred = False
        self.action_taken = False
        self.camera = camera
        self.alarmState = alarmState
        self.event = event
        self.person = person
        self.actions = actions

        self.eventTime = 0

        Alert.alert_count += 1

    def reinitialise(self):
        self.event_occurred = False
        self.action_taken = False

    def set_custom_alertmessage(self,message):
        self.alertString = message

        



