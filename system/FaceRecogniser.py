# FaceRecogniser.
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
import dlib
import sys
import argparse
from PIL import Image
import pickle
import math
import datetime
import threading
import logging
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import time
from operator import itemgetter
from datetime import datetime, timedelta
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
import atexit
from subprocess import Popen, PIPE
import os.path
import numpy as np
import pandas as pd
import aligndlib
import openface

logger = logging.getLogger()

start = time.time()
np.set_printoptions(precision=2)

fileDir = os.path.dirname(os.path.realpath(__file__))
luaDir = os.path.join(fileDir, '..', 'batch-represent')
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

class FaceRecogniser(object):
	"""This class implements face recognition using Openface's
	pretrained neural network and a Linear SVM classifier. Functions 
	below allow a user to retrain the classifier and make predictions 
	on detected faces"""
    
	def __init__(self):
		self.net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,cuda=args.cuda)
		self.align = openface.AlignDlib(args.dlibFacePredictor)
		self.neuralNetLock = threading.Lock()
		self.predictor = dlib.shape_predictor(args.dlibFacePredictor)
		with open("generated-embeddings/classifier.pkl", 'r') as f: # le = labels, clf = classifier
			(self.le, self.clf) = pickle.load(f) # Loads labels and classifier SVM or GMM
    
	def make_prediction(self,rgbFrame,bb):
		"""The function uses the location of a face 
		to detect facial landmarks and perform an affine transform
		to align the eyes and nose to the correct positiion.
		The aligned face is passed through the neural net which
		generates 128 measurements which uniquly identify that face. 
		These measurements are known as an embedding, and are used
		by the classifier to predict the identity of the person"""

		landmarks = self.align.findLandmarks(rgbFrame, bb) 
		if landmarks == None:
			logger.info("//////////////////////  FACE LANDMARKS COULD NOT BE FOUND  //////////////////////////")
			return None
		alignedFace = self.align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE) 

		if alignedFace is None:
		    logger.info("//////////////////////  FACE COULD NOT BE ALIGNED  //////////////////////////")
		    return None

		logger.info("\n//////////////////////  FACE ALIGNED  ////////////////////// \n")
		with self.neuralNetLock :
		    persondict = self.recognize_face(alignedFace)

		if persondict is None:
		    logger.info("\n//////////////////////  FACE COULD NOT BE RECOGNIZED  //////////////////////////\n")
		    return persondict, alignedFace
		else:
		    logger.info("\n//////////////////////  FACE RECOGNIZED  ////////////////////// \n")
		    return persondict, alignedFace

	def recognize_face(self,img):
	    if self.getRep(img) is None:  
	        return None
	    rep1 = self.getRep(img) # Gets embedding representation of image
	    rep = rep1.reshape(1, -1) 
	    start = time.time()
	    predictions = self.clf.predict_proba(rep).ravel() # Computes probabilities of possible outcomes for samples in classifier(clf).
	    maxI = np.argmax(predictions)
	    person1 = self.le.inverse_transform(maxI)
	    confidence1 = int(math.ceil(predictions[maxI]*100))

	    logger.info("Recognition took {} seconds.".format(time.time() - start))
	    logger.info("Recognized {} with {:.2f} confidence.".format(person1, confidence1))

	    persondict = {'name': person1, 'confidence': confidence1, 'rep':rep1}
	    return persondict

	def getRep(self,alignedFace):
	    bgrImg = alignedFace
	    if bgrImg is None:
	        logger.info("unable to load image")
	        return None

	    alignedFace = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	    start = time.time()
	    rep = self.net.forward(alignedFace) # Gets embedding - 128 measurements
	    return rep

	def reloadClassifier(self):
		with open("generated-embeddings/classifier.pkl", 'r') as f: # Reloads character stream from pickle file
			(self.le, self.clf) = pickle.load(f) # Loads labels and classifier SVM or GMM
		return True

	def trainClassifier(self):
		"""Trainng the classifier begins by aligning any images in the 
		training-images directory and putting them into the aligned images
		directory. Each of the aligned face images are passed through the 
		neural net and the resultant embeddings along with their
		labels (names of the people) are used to train the classifier
		which is saved to a pickle file as a character stream"""

		path = fileDir + "/aligned-images/cache.t7" 
		try:
		  os.remove(path) # Remove cache from aligned images folder
		except:
		  logger.info("Tried to remove cache.t7")
		  pass

		start = time.time()
		aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",args.dlibFacePredictor,args.imgDim)
		logger.info("\nAligning images took {} seconds.".format(time.time() - start))
		done = False
		start = time.time()

		done = self.generate_representation()
		   
		if done is True:
		    logger.info("Representation Generation (Classification Model) took {} seconds.".format(time.time() - start))
		    start = time.time()
		    # Train Model
		    self.train("generated-embeddings/","LinearSvm",-1)
		    logger.info("Training took {} seconds.".format(time.time() - start))
		else:
		    logger.info("Generate representation did not return True")


	def generate_representation(self):
		logger.info("\n" + luaDir + "\n")
		self.cmd = ['/usr/bin/env', 'th', os.path.join(luaDir, 'main.lua'),'-outDir',  "generated-embeddings/" , '-data', "aligned-images/"]                 
		if args.cuda:
		    self.cmd.append('-cuda')
		self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)
		outs, errs = self.p.communicate() # Wait for process to exit - wait for subprocess to finish writing to files: labels.csv & reps.csv

		def exitHandler():
		    if self.p.poll() is None:
		        logger.info("<======================Something went Wrong============================>")
		        self.p.kill()
		        return False
		atexit.register(exitHandler) 

		return True


	def train(self,workDir,classifier,ldaDim):
		logger.info("Loading embeddings.")
		fname = "{}/labels.csv".format(workDir) #labels of faces
		labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
		labels = map(itemgetter(1),
		           map(os.path.split,
		               map(os.path.dirname, labels)))  

		fname = "{}/reps.csv".format(workDir) # Representations of faces
		embeddings = pd.read_csv(fname, header=None).as_matrix() # Get embeddings as a matrix from reps.csv
		self.le = LabelEncoder().fit(labels) # LabelEncoder is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1
											 # Fits labels to model
		labelsNum = self.le.transform(labels)
		nClasses = len(self.le.classes_)
		logger.info("Training for {} classes.".format(nClasses))

		if classifier == 'LinearSvm':
		   self.clf = SVC(C=1, kernel='linear', probability=True)
		elif classifier == 'GMM':
		   self.clf = GMM(n_components=nClasses)

		if ldaDim > 0:
		  clf_final =  self.clf
		  self.clf = Pipeline([('lda', LDA(n_components=ldaDim)),
		                  ('clf', clf_final)])

		self.clf.fit(embeddings, labelsNum) #link embeddings to labels

		fName = "{}/classifier.pkl".format(workDir)
		logger.info("Saving classifier to '{}'".format(fName))
		with open(fName, 'w') as f:
		  pickle.dump((self.le,  self.clf), f) # Creates character stream and writes to file to use for recognition

	def getSquaredl2Distance(self,rep1,rep2): 
		"""Returns number between 0-4, Openface calculated the mean between 
		similar faces is 0.99 i.e. returns less than 0.99 if reps both belong 
		to the same person"""

		d = rep1 - rep2
		return np.dot(d, d) 

