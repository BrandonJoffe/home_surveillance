import threading
import time
import numpy as np
import cv2
import ImageProcessor

CAPTURE_HZ = 15.0

class VideoCamera(object):
    def __init__(self,camURL):
	 	self.video = cv2.VideoCapture(camURL)
	 	self.url = camURL
		#if not self.video.isOpened():
		#	self.video.open()
		# Start a thread to continuously capture frames.
		self.capture_frame = None
		# Use a lock to prevent access concurrent access to the camera.
		self.capture_lock = threading.Lock()
		self.capture_thread = threading.Thread(target=self.get_frame)
		self.capture_thread.daemon = True
		self.capture_thread.start()
		self.people = []
		print("Loading Stream From IP Camera ",camURL)
       
    def __del__(self):
        self.video.release()
    	
    def get_frame(self):
		while True:
		        success, frame = self.video.read()
			with self.capture_lock:
				self.capture_frame = None
				if success:
					self.capture_frame = frame
			time.sleep(1.0/CAPTURE_HZ)

	# def to_bytes(n, length, endianess='big'):
	#     h = '%x' % n
	#     s = ('0'*(len(h) % 2) + h).zfill(length*2).decode('hex')
	#     return s if endianess == 'big' else s[::-1]

    def read(self):
		"""Read a single frame from the camera and return the data as an OpenCV
		image (which is a numpy array).
		"""	
		frame = None
		with self.capture_lock:
			frame = self.capture_frame
		# If there are problems, keep retrying until an image can be read.
		while frame == None:
			time.sleep(0)
			with self.capture_lock:
				frame = self.capture_frame
		# Save captured image for debugging.
		#cv2.imwrite("debug.png", frame)
		# Return the capture image data.
		height, width, channels = frame.shape
		frame = cv2.resize(frame,(480, 320), interpolation = cv2.INTER_CUBIC)
		#Do Frame Processing
		#//////////////////////////////////////////////////////////////////




		height, width, channels = frame.shape
		processed_image = ImageProcessor.detect_faces(self,frame,width,height)
		






		#//////////////////////////////////////////////////////////////////

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
		ret, jpeg = cv2.imencode('.jpg', processed_image)
		return jpeg.tostring()



class Person(object):
    person_count = 0

    def __init__(self,personCoord):
        
        self.personCoord = personCoord
        self.identity = "unknown_" + str(Person.person_count)
        self.confidence = 0
        Person.person_count += 1 
      	#self.tracker = dlib.correlation_tracker()
    
    def get_identity(self):
        return self.identity

    def set_identity(self, id):
        self.identity = id

    def recognize_face(self):     
       return

#     def update_position(self, newCoord):
#        self.personCoord = newCoord

#     def get_current_position(self):
#        return personCoord
 #def start_tracking(self,img):
    
    #def update_tracker(self,img):    

    #def find_face(self):     
    #   return
