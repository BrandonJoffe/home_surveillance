# Home Surveilance with Facial Recognition. 

The purpose of this project was to develop a low-cost, adaptive and extensible surveillance system which focused on identifying and alerting for potential home intruders. It can integrate into an existing alarm system and provides customizable alerts for the user. It can process several IP cameras and can distinguish between someone who is in the face database and someone who is not (a potential intruder). This is by no means a finished product. It was developed for my thesis, and there is so much that can be done to make this an incredibly useful tool.

---

[![solarized dualmode](https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/prototype/system/debugging/dashboard.png)](#features)

## System Overview ##

### Whats inside? ###

The main system components include a dedicated system server which performs all the central processing and web based communication and a Raspberry PI which hosts the alarm control interface.  

[![solarized dualmode]
(https://github.com/BrandonJoffe/home_surveillance/blob/prototype/system/debugging/designOverview-2.png?raw=true)](#features)

### How does it work? ###

The SurveillanceSystem object is the heart of the system. It can process several IPCameras and monitors the system's alerts. A FaceRecogniser object provides functions for training a linear SVM classifier using the face database and includes all the functions necessary to perform face recognition using Openface's pre-trained neural network (thank you Brandon Amos!!). The IPcamera object streams frames directly from an IP camera and makes them available for processing, and streaming to the web client. Each IPCamera has its own MotionDetector and FaceDetector object, which are used by other subsequent processes to perform face recognition and person tracking. The FlaskSocketIO object streams jpeg frames (mjpeg) to the client and transfers JSON data using HTTP POST requests and web sockets. Finally, the flask object on the Raspberry PI simply controls a GPIO interface which can be directly connected to an existing wired alarm panel.
 
[![solarized dualmode]
(https://github.com/BrandonJoffe/home_surveillance/blob/master/system/testing/implementation/finalSystemImplementation.png?raw=true?raw=true)](#features)

### How do I setup the network? ###

How the network is setup is really up to you, I used a PoE switch to connect all my IP cameras to the network, you can stream from cameras that are directly connected to the NVR, and there are many other ways to setup the network.

[![solarized dualmode]
(https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/master/system/testing/implementation/testingEnvironment.png?raw=true)](#features)

## Facial Recognition Accuracy ##

The graph below shows the recognition accuracy of identifying known and unknown people with the use of an unknown class in the classifier and an unknown confidence threshold. Currently, Openface has an accuracy of 0.9292 ± 0.0134 on the LFW benchmark, and although benchmarks are great for comparing the accuracy of different techniques and algorithms, they do not model a real world surveillance environment. The tests conducted were taken in a home surveillance scenario with two different IP cameras in an indoor and outdoor environment at different times of the day. A total of 15 people were recorded an captured to train the classifier. Face images were also taken from both the LFW database as well as the FEI database, to test the recognition accuracy of identifying unknown people and create the unknown class in the classifier. 

[![solarized dualmode]
(https://github.com/BrandonJoffe/home_surveillance/blob/master/system/testing/implementation/RecognitionAccuracy.png?raw=true)](#features)

At an unknown confidence threshold of 20, the recognition accuracy of identifying an unknown person is 81.25%, while the accuracy of identifying a known person is 75.52%. This produces a final combined system recognition accuracy of 78.39%. 
 
## System Proccessing Capability ##

The systems ability to process several cameras simultaneously in real time with different resolutions is shown in the graph below. These tests were conducted on a 2011 Mac Book Pro running Yosemite. 

[![solarized dualmode]
(https://github.com/BrandonJoffe/home_surveillance/blob/master/system/testing/implementation/processingCapability.png?raw=true)](#features)

By default, the SurveillanceSystem object resizes frames to a ratio where the height is always 640 pixels. This was chosen as it produced the best results with regards to its effects on processing and face recognition accuracy. Although the graph shows the ability of the system to process up to 9 cameras in real time using a resolution of 640x480, it cannot stream 9 cameras to the web client simultaneously with the approach currently being used. During testing up to 6 cameras were able to stream in real time, but this was not always the case. The most consistent real-time streaming included the use of only three cameras.

## Installation and Usage ##

### Docker ###

Openface provides an automated docker build which works well on Ubuntu and OSX (Haven't attempted Windows) and was used with the addition of a few Flask dependencies for development. Docker for MAC often gave a bad response from the Docker engine which is currently an unsolved problem. If you would like to deploy the system, Docker for MAC is currently not a viable solution. Other than that, any Ubuntu OS running a 64-bit x86 architecture should run the Docker container without any problems. To get access to your application outside your home network, you'll have to open up port 5000 on your router and assign the IP address of your system server

---

1) Pull Docker Image

```
docker pull bjoffe/openface_flask_v2
```

2) Run Docker image, make sure you mount your User (for MAC) or home (for Ubuntu) directory as a volume so you can access your local files

```
docker run -v /Users/:/host -p 9000:9000 -p 8000:8000 -p 5000:5000 -t -i bjoffe/openface_flask_v2  /bin/bash

```

- Navigate to the home_surveillance project inside the volume within your Docker container
- Move into the system directory

```
cd system
```
- Run WebApp.py
```
python WebApp.py
```
- Visit ```localhost:5000 ```
- Login Username: ```admin``` Password ```admin```

## Notes and Features ##

>### *Camera Settings*
- To add your own IP camera simply add the URL of the camera into field on the camera panel and choose 1 out of the 5 proccessing settings and your prefered face detection method. 
- Unfortunately I haven't included a means to remove the cameras once they have been added, however, this will be added shortly.

>### *Customizable Alerts*
- The Dashboard allows you to configure your own email and alarm trigger alerts. 
- The alerts panel allows you to set up certain events such as the recognition of a particular person or motion detection so    that you receive an email alert when the event occurs. The confidence slider sets the accuracy that you would like to use for recognition events. By default you'll receive a notification if a person is recognised with a percentage greater than 50%.
- The alarm control panel sends http post requests to a web server on a Raspberry PI to control GPIO pins.

>### *Face Recognition and the Face Database*
- Faces that are detected are shown in the faces detected panel on the Dashboard
- There is currently no formal database setup and the faces are stored in the aligned-images & training-images directories
- To add faces to the database add a folder of images with the name of the person and retrain the classifier by selecting the retrain database on the client dashboard. Images can also be added through the dashboard but can currently only be added one at a time.
- To perform accurate face recognition, twenty or more face images should be used. Furthermore, images taken in the surveillance enviroment (i.e use the IP cameras to capture face images - this can be acheived by using the face_capture option in the SurveillanceSystem script and creating your own face directory) produce better results as a posed to adding images taken else where.
- A person is classified as unknown if they are recognised with a confidence lower than 20% or are predicted as unknown by the classifier

>### *Security*
- Unfortunately, the only security that has been implemented includes basic session management and hard coded authenetication. Where each user is faced with a login page. Data and password encryption is a feature for future developement.

>### *Some Issues and Limitations*
- Accasionaly Flask disconnects and this causes the video streaming to break. Fixing this may involve using another web framework. However, it could also be fixed by adding a camera reload function which will be added shortly. 
- Currently the tracking algorithm is highly dependent upon the accuracy of the background model generated by the MotionDetector object. The tracking algorithm is based on a background subtraction approach, and if the camera is placed in an outdoor enviroment where there is likely to be moving trees, vast changes in lighting etc it may not beable work effectly.
- Both Dlib's and OpenCV's face detection methods produce false positives now and again. The system does incorporate some mitigation for these false detections by using more rigourous parameters, and background subtraction to ignore any detections that occur outside the region of interest.
- The more people and face images you have in the database the longer it takes to train the classifier, it may take up to several minutes. Accasionaly Docker for MAC killed the python process used for training, in which case you have to start all over again.

## Ideas for Future developement ##

- Database Implementation

- Improved Security

- Open set recognition for accurate identification of unknown people

- Behaviour recognition using neural networks

- Optimizing motion detection and tracking algorithms

- Intergration with third party services such as facebook to recognise your friends

- The addition of home automation control features 

and many more...

# License
---

Copyright 2016, Brandon Joffe, All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

- http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# References
---

- Video Streaming Code - http://www.chioka.in/python-live-video-streaming-example/
- Flask Web Server GPIO - http://mattrichardson.com/Raspberry-Pi-Flask/
- Openface Project - https://cmusatyalab.github.io/openface/
- Flask Websockets - http://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent

 


 
