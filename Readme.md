# Home Surveilance with Facial Recognition. 
---
[![solarized dualmode](https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/prototype/system/debugging/dashboard.png)](#features)

## System Overview ##

### Whats Inside? ###


[![solarized dualmode]
(https://github.com/BrandonJoffe/home_surveillance/blob/prototype/system/debugging/designOverview-2.png?raw=true)](#features)

### How Does It Work? ###

[![solarized dualmode]
(https://github.com/BrandonJoffe/home_surveillance/blob/master/system/testing/implementation/finalSystemImplementation.png?raw=true?raw=true)](#features)

### How Do I Setup The Network? ###

[![solarized dualmode]
(https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/master/system/testing/implementation/testingEnvironment.png?raw=true)](#features)

## Installation and Usage ##

1) Pull Docker Image

```
docker pull bjoffe/openface_flask_v2
```

2) Run Docker image, make sure you mount your User directory as a volume so you can access your local files

```
docker run -v /Users/:/host/Users -p 9000:9000 -p 8000:8000 -p 5000:5000 -t -i bjoffe/openface_flask_v2  /bin/bash

```

- Navigate to the home_surveillance project inside the Docker container
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

## Recognition Accuracy ##

## Notes and Features ##
---

- To add your own IP camera simply add the URL of the camera into field on the camera panel on the client dashboard. 

- Cameras can be set to perform motion detection, face recognition and person tracking

- Faces that are detected are shown in the faces detected panel on the Dashboard

- There is currently no formal database setup and the faces are stored in the aligned-images & training-images directories

- To add faces to the database add a folder of images with the name of the person and retrain the classifier by selecting the retrain database on the client dashboard. Images can also be added through the dashboard but can currently only be added one at a time.

- The Dashboard allows you to configure your own email and alarm trigger alerts. 

- The alerts panel allows you to set up certain events such as the recognition of a particular person or motion detection so that you receive an email alert when the event occurs. The confidence slider sets the accuracy that you would like to use for recognition events. By default you'll receive a notification if a person is recognised with a percentage greater than 50%.

- A person is classified as unknown if they are identified with a confidence less than 50%

- The alarm control panel sends http post requests to a web server on a Raspberry PI to control GPIO pins. The RPI alarm interface code is yet to be uploaded, but will be available shortly.

- On a 2011 mac book pro the system can process up to 6 cameras however the video streaming does lag. That being said, frames are still processed in real time on the server.

- Currently there are a few bugs and the code isn't fully commented.

- There is plenty work to be done for this system to become a fully functional and secure open source home surveillance system.

- This project is being developed for the purpose of my thesis and I hope to have a fully functional system by the end of October 2016.

# License
---

Copyright 2016, Brandon Joffe, All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# References
---

- Video Streaming Code - http://www.chioka.in/python-live-video-streaming-example/
- Flask Web Server GPIO - http://mattrichardson.com/Raspberry-Pi-Flask/
- Openface Project - https://cmusatyalab.github.io/openface/
- Flask Websockets - http://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent

 


 
