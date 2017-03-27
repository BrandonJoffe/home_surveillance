# Camera Class
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

import threading
import time
import numpy as np
import cv2
import cv2.cv as cv
import ImageUtils
import dlib
import openface
import os
import argparse
import logging
import SurveillanceSystem
import MotionDetector
import FaceDetector

#logging.basicConfig(level=logging.DEBUG,
#                    format='(%(threadName)-10s) %(message)s',
#                    )

logger = logging.getLogger(__name__)

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

CAPTURE_HZ = 30.0 # Determines frame rate at which frames are captured from IP camera

class IPCamera(object):
	"""The IPCamera object continually captures frames
	from a camera and makes these frames available for
	proccessing and streamimg to the web client. A 
	IPCamera can be processed using 5 different processing 
	functions detect_motion, detect_recognise, 
	motion_detect_recognise, segment_detect_recognise, 
	detect_recognise_track. These can be found in the 
	SureveillanceSystem object, within the process_frame function"""

	def __init__(self,camURL, cameraFunction, dlibDetection, fpsTweak):
		logger.info("Loading Stream From IP Camera: " + camURL)
		self.motionDetector = MotionDetector.MotionDetector()
		self.faceDetector = FaceDetector.FaceDetector()
		self.processing_frame = None
		self.tempFrame = None
		self.captureFrame  = None
		self.streamingFPS = 0 # Streaming frame rate per second
		self.processingFPS = 0
		self.FPSstart = time.time()
		self.FPScount = 0
		self.motion = False # Used for alerts and transistion between system states i.e from motion detection to face detection
		self.people = {} # Holds person ID and corresponding person object 
		self.trackers = [] # Holds all alive trackers
		self.cameraFunction = cameraFunction 
		self.dlibDetection = dlibDetection # Used to choose detection method for camera (dlib - True vs opencv - False)
		self.fpsTweak = fpsTweak # used to know if we should apply the FPS work around when you have many cameras
		self.rgbFrame = None
		self.faceBoxes = None
		self.captureEvent = threading.Event()
		self.captureEvent.set()
		self.peopleDictLock = threading.Lock() # Used to block concurrent access to people dictionary
		self.video = cv2.VideoCapture(camURL) # VideoCapture object used to capture frames from IP camera
		logger.info("We are opening the video feed.")
	 	self.url = camURL
		if not self.video.isOpened():
			self.video.open()
		logger.info("Video feed open.")
		self.dump_video_info()  # logging every specs of the video feed
		# Start a thread to continuously capture frames.
		# The capture thread ensures the frames being processed are up to date and are not old
		self.captureLock = threading.Lock() # Sometimes used to prevent concurrent access
		self.captureThread = threading.Thread(name='video_captureThread',target=self.get_frame)
		self.captureThread.daemon = True
		self.captureThread.start()
		self.captureThread.stop = False

	def __del__(self):
		self.video.release()

	def get_frame(self):
		logger.debug('Getting Frames')
		FPScount = 0
		warmup = 0
		#fpsTweak = 0  # set that to 1 if you want to enable Brandon's fps tweak. that break most video feeds so recommend not to
		FPSstart = time.time()

		while True:
			success, frame = self.video.read()
			self.captureEvent.clear() 
			if success:		
				self.captureFrame  = frame
				self.captureEvent.set() 

			FPScount += 1 

			if FPScount == 5:
				self.streamingFPS = 5/(time.time() - FPSstart)
				FPSstart = time.time()
				FPScount = 0

			if self.fpsTweak:
				if self.streamingFPS != 0:  # If frame rate gets too fast slow it down, if it gets too slow speed it up
					if self.streamingFPS > CAPTURE_HZ:
						time.sleep(1/CAPTURE_HZ)
					else:
						time.sleep(self.streamingFPS/(CAPTURE_HZ*CAPTURE_HZ))

	def read_jpg(self):
		"""We are using Motion JPEG, and OpenCV captures raw images,
		so we must encode it into JPEG in order to stream frames to
		the client. It is nessacery to make the image smaller to
		improve streaming performance"""

		capture_blocker = self.captureEvent.wait()  
		frame = self.captureFrame 	
		frame = ImageUtils.resize_mjpeg(frame)
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tostring()

	def read_frame(self):
		capture_blocker = self.captureEvent.wait()  
		frame = self.captureFrame 	
		return frame

	def read_processed(self):
		frame = None
		with self.captureLock:
			frame = self.processing_frame	
		while frame == None: # If there are problems, keep retrying until an image can be read.
			with self.captureLock:	
				frame = self.processing_frame

		frame = ImageUtils.resize_mjpeg(frame)
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tostring()

	def dump_video_info(self):
		logger.info("---------Dumping video feed info---------------------")
		logger.info("Position of the video file in milliseconds or video capture timestamp: ")
		logger.info(self.video.get(cv.CV_CAP_PROP_POS_MSEC))
		logger.info("0-based index of the frame to be decoded/captured next: ")
		logger.info(self.video.get(cv.CV_CAP_PROP_POS_FRAMES))
		logger.info("Relative position of the video file: 0 - start of the film, 1 - end of the film: ")
		logger.info(self.video.get(cv.CV_CAP_PROP_POS_AVI_RATIO))
		logger.info("Width of the frames in the video stream: ")
		logger.info(self.video.get(cv.CV_CAP_PROP_FRAME_WIDTH))
		logger.info("Height of the frames in the video stream: ")
		logger.info(self.video.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
		logger.info("Frame rate:")
		logger.info(self.video.get(cv.CV_CAP_PROP_FPS))
		logger.info("4-character code of codec.")
		logger.info(self.video.get(cv.CV_CAP_PROP_FOURCC))
		logger.info("Number of frames in the video file.")
		logger.info(self.video.get(cv.CV_CAP_PROP_FRAME_COUNT))
		logger.info("Format of the Mat objects returned by retrieve() .")
		logger.info(self.video.get(cv.CV_CAP_PROP_FORMAT))
		logger.info("Backend-specific value indicating the current capture mode.")
		logger.info(self.video.get(cv.CV_CAP_PROP_MODE))
		logger.info("Brightness of the image (only for cameras).")
		logger.info(self.video.get(cv.CV_CAP_PROP_BRIGHTNESS))
		logger.info("Contrast of the image (only for cameras).")
		logger.info(self.video.get(cv.CV_CAP_PROP_CONTRAST))
		logger.info("Saturation of the image (only for cameras).")
		logger.info(self.video.get(cv.CV_CAP_PROP_SATURATION))
		logger.info("Hue of the image (only for cameras).")
		logger.info(self.video.get(cv.CV_CAP_PROP_HUE))
		logger.info("Gain of the image (only for cameras).")
		logger.info(self.video.get(cv.CV_CAP_PROP_GAIN))
		logger.info("Exposure (only for cameras).")
		logger.info(self.video.get(cv.CV_CAP_PROP_EXPOSURE))
		logger.info("Boolean flags indicating whether images should be converted to RGB.")
		logger.info(self.video.get(cv.CV_CAP_PROP_CONVERT_RGB))
		logger.info("--------------------------End of video feed info---------------------")