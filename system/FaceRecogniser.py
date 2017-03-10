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

logger = logging.getLogger(__name__)

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

        logger.info("Opening classifier.pkl to load existing known faces db")
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
            logger.info("///  FACE LANDMARKS COULD NOT BE FOUND  ///")
            return None
        alignedFace = self.align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            logger.info("///  FACE COULD NOT BE ALIGNED  ///")
            return None

        logger.info("////  FACE ALIGNED  // ")
        with self.neuralNetLock :
            persondict = self.recognize_face(alignedFace)

        if persondict is None:
            logger.info("/////  FACE COULD NOT BE RECOGNIZED  //")
            return persondict, alignedFace
        else:
            logger.info("/////  FACE RECOGNIZED  /// ")
            return persondict, alignedFace

    def recognize_face(self,img):
        if self.getRep(img) is None:
            return None
        rep1 = self.getRep(img) # Gets embedding representation of image
        logger.info("Embedding returned. Reshaping the image and flatting it out in a 1 dimension array.")
        rep = rep1.reshape(1, -1)   #take the image and  reshape the image array to a single line instead of 2 dimensionals
        start = time.time()
        logger.info("Submitting array for prediction.")
        predictions = self.clf.predict_proba(rep).ravel() # Computes probabilities of possible outcomes for samples in classifier(clf).
        #logger.info("We need to dig here to know why the probability are not right.")
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
            logger.error("unable to load image")
            return None

        logger.info("Tweaking the face color ")
        alignedFace = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        start = time.time()
        logger.info("Getting embedding for the face")
        rep = self.net.forward(alignedFace) # Gets embedding - 128 measurements
        return rep

    def reloadClassifier(self):
        with open("generated-embeddings/classifier.pkl", 'r') as f: # Reloads character stream from pickle file
            (self.le, self.clf) = pickle.load(f) # Loads labels and classifier SVM or GMM
        logger.info("reloadClassifier called")
        return True

    def trainClassifier(self):
        """Trainng the classifier begins by aligning any images in the
        training-images directory and putting them into the aligned images
        directory. Each of the aligned face images are passed through the
        neural net and the resultant embeddings along with their
        labels (names of the people) are used to train the classifier
        which is saved to a pickle file as a character stream"""

        logger.info("trainClassifier called")

        path = fileDir + "/aligned-images/cache.t7"
        try:
            os.remove(path) # Remove cache from aligned images folder
        except:
            logger.info("Failed to remove cache.t7")
            pass

        logger.info("Succesfully removed " + path)
        start = time.time()
        aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",args.dlibFacePredictor,args.imgDim)
        logger.info("Aligning images for training took {} seconds.".format(time.time() - start))
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
        logger.info("lua Directory:    " + luaDir)
        self.cmd = ['/usr/bin/env', 'th', os.path.join(luaDir, 'main.lua'),'-outDir',  "generated-embeddings/" , '-data', "aligned-images/"]
        logger.info("lua command:    " + str(self.cmd))
        if args.cuda:
            self.cmd.append('-cuda')
            logger.info("using -cuda")
        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)
        outs, errs = self.p.communicate() # Wait for process to exit - wait for subprocess to finish writing to files: labels.csv & reps.csv
        logger.info("Waiting for process to exit to finish writing labels and reps.csv")

        def exitHandler():
            if self.p.poll() is None:
                logger.info("<=Something went Wrong===>")
                self.p.kill()
                return False
        atexit.register(exitHandler)

        return True


    def train(self,workDir,classifier,ldaDim):
        logger.info("Loading embeddings.")
        fname = "{}/labels.csv".format(workDir) #labels of faces
        if os.stat(fname).st_size > 0:
            logger.info(fname + " file is not empty")
            labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
        else:
            logger.info(fname + " file is empty")
            labels = {1:"aligned-images/mathieu/1.png"}
        labels = map(itemgetter(1),
            map(os.path.split,
            map(os.path.dirname, labels)))

        fname = "{}/reps.csv".format(workDir) # Representations of faces
        if os.stat(fname).st_size > 0:
            logger.info(fname + " file is not empty")
            embeddings = pd.read_csv(fname, header=None).as_matrix() # Get embeddings as a matrix from reps.csv
        else:
            logger.info(fname + " file is empty")
            embeddings = {-0.064819745719433,0.086162053048611,0.11779563874006,-0.067650333046913,0.0036357014905661,-0.0070062829181552,-0.036009397357702,-0.011870233342052,-0.12318318337202,0.035625800490379,0.038776393979788,0.053646054118872,-0.016345497220755,0.11617664247751,0.11193278431892,-0.1157177016139,-0.021303432062268,-0.061479087918997,0.17499601840973,0.048021901398897,-0.046290867030621,0.069292806088924,0.050160840153694,0.028962032869458,0.033040937036276,0.087781712412834,0.13426727056503,0.075119271874428,0.072730585932732,0.018530936911702,-0.086269184947014,-0.2181206792593,0.18646237254143,0.021509036421776,0.044785920530558,0.026634698733687,0.12029408663511,0.096228748559952,-0.019187683239579,-0.011889624409378,0.030606608837843,-0.012100863270462,0.11255705356598,0.020040338858962,-0.2232770472765,0.060600552707911,0.13450945913792,-0.078542776405811,-0.067810088396072,0.034219156950712,0.0044375983998179,0.049074523150921,-0.049328397959471,0.11790949851274,0.20804460346699,0.072516836225986,-0.011820805259049,0.047062665224075,-0.10259047150612,-0.034201834350824,-0.099936619400978,-0.093768760561943,0.077673763036728,-0.033444426953793,0.14740560948849,0.07337848842144,-0.031113782897592,-0.052998770028353,-0.05852147564292,0.097891844809055,0.017302541062236,-0.007969313301146,-0.037028376013041,0.021780924871564,0.22088295221329,0.05962897092104,-0.01658895239234,0.073803663253784,0.11678933352232,-0.058586984872818,-0.076783291995525,0.0011724757496268,0.14104007184505,0.11061990261078,0.051374554634094,-0.1166664659977,0.020046267658472,0.099389806389809,0.083941854536533,0.12955328822136,0.018088771030307,-0.08964940905571,0.060039047151804,0.022854413837194,-0.034995667636395,-0.19448159635067,-0.00080694549251348,0.13129590451717,0.087584123015404,-0.085186399519444,-0.085907638072968,0.14244784414768,0.066603802144527,-0.076008372008801,-0.0840705037117,0.11509071290493,0.039160214364529,-0.023864150047302,0.05813355371356,-0.12844789028168,0.06084568053484,-0.0013438333990052,0.0028949021361768,0.061773918569088,0.0091053172945976,-0.016758112236857,0.099529922008514,-0.10636233538389,-0.02447871118784,0.093821138143539,0.099414013326168,-0.074449948966503,-0.048774160444736,-0.040542088449001,0.010523811914027,0.12757521867752,0.20010431110859,0.11064236611128}

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

