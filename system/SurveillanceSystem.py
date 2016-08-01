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

import Camera
import openface
import aligndlib

#Get paths for models
#//////////////////////////////////////////////////////////////////////////////////////////////
start = time.time()
np.set_printoptions(precision=2)



#///////////////////////////////////////////////////////////////////////////////////////////////



                              
class Surveillance_System(object):
    
   def __init__(self):
        self.training = True
        self.cameras = []
        self.svm = None


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

        #Build Classification Model
        self.build_classification_model(luaDir)

        #default IP cam
        self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("Test.mov"))

   def initialize(self):
        return
        #do training or get trained file
        #open cameras with Threads
        #Get Notification Details
        #Start processing

   def build_classification_model(self,fileDir):
        #1 align raw images   
        aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",self.args.dlibFacePredictor,self.args.imgDim)

        #2 Generate Representation ./batch-represent/main.lua -outDir <feature-directory> -data <path-to-aligned-data> creates reps.csv and labels.csv in <feature-directory>.        
        self.cmd = ['/usr/bin/env', 'th', os.path.join(fileDir, 'main.lua'),'-outDir',  "generated-embeddings/" , '-data', "aligned-images/"]
                    
        if self.args.cuda:
            self.cmd.append('-cuda')
        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)

        def exitHandler():
            if self.p.poll() is None:
                self.p.kill()
        atexit.register(exitHandler) 

        #3 Create Classification Model ./demos/classifier.py train <feature-directory>
        self.train("generated-embeddings/","LinearSvm",-1)
       

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