# Home Surveilance with Facial Recognition. 
---

# Installation
---

1) Pull Docker Image

```
docker pull bjoffe/openface_flask
```

2) Run Docker image, make sure you mount your User directory as a volume so you can access your local files

```
docker run -v /Users/:/host/Users -p 9000:9000 -p 8000:8000 -p 5000:5000 -t -i bjoffe/openface_flask  /bin/bash

```

# Usage
---

- Navigate to the home_surveillance project inside the Docker container
- Move into the system directory
```
cd system
```
- Run WebSocket.py
```
python WebSocket.py
```
- Visit ```localhost:5000 ```
- Login Username: ```admin``` Password ```admin```

# Notes and Features
---

- By default the system processes a single video
- To add your own IP camera simply add the URL of the camera to the SurveillanceSystem.py script within the Surveillance_System constructer method. ``` self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg")) ``` 
- Faces that are detected are shown in the faces detected panel on the Home Surveillance Dashboard
- The Dashboard allows you to configure your own email and push notification alerts (You'll need to download the instapush app and add your email address in the SurveillanceSystem.py script).
- This project is being developed for the purpose of my thesis and I hope to have a fully functional system by the end of October 2016.
- Currently there are a few bugs and the code is not well commented.

# References
---

Video Streaming Code - http://www.chioka.in/python-live-video-streaming-example/

 
