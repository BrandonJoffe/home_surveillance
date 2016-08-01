# Face Surveillance
---

Web Application for Home surveilance with facial recognition. 

# Installation
---

1) Pull Openface Docker Image

2) Run The Docker Image

4) Install Flask-Socket dependencies from https://github.com/miguelgrinberg/Flask-SocketIO using setup.py and requirements.txt

5) Create new image from Docker container

6) Run Docker image, make sure you mount your User directory as a volume so you can access your local files

```
docker run -v /Users:/host/Users -p 9000:9000 -p 8000:8000 -p 5000:5000 -t -i bamos/openface /bin/bash
```

# Usage
---

Work In Progress...

# Notes
---

# References
---

Video Streaming Code - http://www.chioka.in/python-live-video-streaming-example/

 
