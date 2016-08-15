# Home Surveillance With Facial Recognition
---

Web Application for Home surveilance with facial recognition. 

# Installation
---

1) Pull openface_flask Docker Image

```
docker pull bjoffe/openface_flask
```

6) Run Docker image, make sure you mount your User directory as a volume so you can access your local files

```
docker run -v /Users:/host/Users -p 9000:9000 -p 8000:8000 -p 5000:5000 -t -i bjoffe/openface_flask  /bin/bash
```

# Usage
---

```
cd system
python WebSocket.py
```
visit localhost:5000 in your web browser

# Notes
---
- By default the system processes three videos (high resolution - there is a noticable lag)
- IP cameras can be used by simply going into the the SurveillanceSystem Script and adding the IP camera URLs to the Class constructor self.cameras.append(Camera.VideoCamera("CamURL"))
- Please be aware that this project is far from finished and certainly does contain bugs!!
- This code is being developed for the purpose of my thesis and will hopefully be ready to be shared for everybody to enjoy by October 2016
# References
---

Video Streaming Code - http://www.chioka.in/python-live-video-streaming-example/

 
