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

import Camera
import openface
import aligndlib
import ImageProcessor

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
        self.cameras.append(Camera.VideoCamera("debugging/Test.mov"))
        self.cameras.append(Camera.VideoCamera("debugging/Test.mov"))
        self.cameras.append(Camera.VideoCamera("debugging/Test.mov"))
        #self.cameras.append(Camera.VideoCamera("debugging/example_01.mp4"))

        # processing frame thread
        for i, cam in enumerate(self.cameras):
          self.proccesing_lock = threading.Lock()
          thread = threading.Thread(name='process_thread_' + str(i),target=self.process_frame,args=(cam,))
          thread.daemon = False
          self.camera_threads.append(thread)
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
        if camera.capture_frame is not None:      
          height, width, channels = camera.capture_frame.shape
          frame = ImageProcessor.detect_faces(camera,camera.capture_frame,width,height)
          camera.processed_frame = frame

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


   def add_face():
      return

   def add_notifications():
      return

   # def camera_stream(self,camera):
   #    while True:
   #       logging.debug('reading')
   #       camera.processed_frame = camera.read()

   # Ensures all scripts use same instance of surveillance object

# if __name__ == "__main__":

#   Home_surveillance = Surveillance_System.getInstance()
#   WebSocket.start()


  

 




