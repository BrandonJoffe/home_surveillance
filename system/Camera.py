import threading
import time
import numpy as np
import cv2
import ImageProcessor
import dlib
import openface
import os
import argparse

import logging

CAPTURE_HZ = 5.0

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )


class VideoCamera(object):
    def __init__(self,camURL):
		print("Loading Stream From IP Camera ",camURL)

		self.previous_frame = None
		self.processed_frame = None
		self.capture_frame = None
		self.firstFrame = None
		self.people = []

	 	self.video = cv2.VideoCapture(camURL)
	 	self.url = camURL
		if not self.video.isOpened():
			self.video.open()
		# Start a thread to continuously capture frames.
		# Use a lock to prevent access concurrent access to the camera.
		self.capture_lock = threading.Lock()
		self.capture_thread = threading.Thread(name='video_capture_thread',target=self.get_frame)
		self.capture_thread.daemon = True
		self.capture_thread.start()

		#Neural Net Object
		fileDir = os.path.dirname(os.path.realpath(__file__))
		modelDir = os.path.join(fileDir, '..', 'models')
		dlibModelDir = os.path.join(modelDir, 'dlib')
		openfaceModelDir = os.path.join(modelDir, 'openface')
		parser = argparse.ArgumentParser()
		parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
		parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
		parser.add_argument('--cuda', action='store_true')
		args = parser.parse_args()
		self.net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,cuda=args.cuda)
                               
		
       
    def __del__(self):
        self.video.release()
    	
    def get_frame(self):
		logging.debug('Getting Frames')
		while True:
		        success, frame = self.video.read()
			with self.capture_lock:
				self.capture_frame = None
				if success:	
					r = 480.0 / frame.shape[1]
					dim = (480, int(frame.shape[0] * r))
					# perform the actual resizing of the image and show it
					frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)		
					self.capture_frame = frame
			time.sleep(1.0/CAPTURE_HZ)

    def read(self):

 		#Block until processed_frame is available
 		while self.processed_frame == None:
 			continue
			
		frame = self.processed_frame 	

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
		ret, jpeg = cv2.imencode('.jpg', frame)
		#cv2.imwrite("stream/frame.jpg", frame)
		self.previous_frame = frame
		return jpeg.tostring()



class Person(object):
    person_count = 0

    def __init__(self,personCoord):
        
        self.personCoord = personCoord
        self.identity = "unknown_" + str(Person.person_count)
        self.confidence = 0
        Person.person_count += 1 
      	self.tracker = dlib.correlation_tracker()
    
    def get_identity(self):
        return self.identity

    def set_identity(self, id):
        self.identity = id

    def recognize_face(self):     
       return

    def update_position(self, newCoord):
       self.personCoord = newCoord

    def get_current_position(self):
       return self.personCoord

    def start_tracking(self,img):
 	   self.tracker.start_track(img, dlib.rectangle(self.personCoord))
    
    def update_tracker(self,img):    
       self.tracker.update(img)

    def get_position(self):
       return self.tracker.get_position()

    def find_face(self):     
       return



   # tracking = FaceTracking(detect_min_size=detect_min_size,
   #                          detect_every=detect_every,
   #                          track_min_overlap_ratio=track_min_overlap_ratio,
   #                          track_min_confidence=track_min_confidence,
   #                          track_max_gap=track_max_gap)

