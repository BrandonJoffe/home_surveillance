# Web Socket Server
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


# main.py
#from gevent import monkey
#monkey.patch_all()
#import redis
from flask import Flask, render_template, Response, redirect, url_for, request, jsonify
import Camera
from flask.ext.socketio import SocketIO,send, emit #Socketio depends on gevent
import SurveillanceSystem
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

Home_Surveillance = SurveillanceSystem.Surveillance_System()


@app.route('/', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid username or password. Please try again'
        else:
            return redirect(url_for('home'))

    return render_template('login.html', error = error)

@app.route('/home')
def home():
    return render_template('index.html')

 
def gen(camera):
    while True:
        frame = camera.read_processed()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # builds 'packet' of data with header and payload 

@app.route('/video_feed_one')
def video_feed_one():

    return Response(gen(Home_Surveillance.cameras[0]),
                    mimetype='multipart/x-mixed-replace; boundary=frame') # a stream where each part replaces the previous part the multipart/x-mixed-replace content type must be used.

# @app.route('/video_feed_two')
# def video_feed_two():
#     return Response(gen(Home_Surveillance.cameras[0]),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed_three')
# def video_feed_three():
#     return Response(gen(Home_Surveillance.cameras[2]),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# def get_faceimg(camera,name):
#     img = camera.people[name].thumbnail
#     return (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n') 
               

# def get_facedata(cameras):
#     peopledata = []
#     persondict = {}
#     for camera in cameras:
#         for key, obj in camera.people.iteritems():
#             persondict = {'name': key , 'confidence': obj.confidence}
#             peopledata.append(persondict)
#     return peopledata


# @socketio.on('face_data', namespace='/test') 
# def handle_my_custom_event():
    
#     return Response(json.dumps(get_faces(Home_Surveillance.cameras)), content_type='application/json') 

@socketio.on('my event', namespace='/test') #socketio used to receive websocket messages # Namespaces allow a cliet to open multiple connectiosn to the server that are multiplexed on a single socket
def test_message(message):   #custom events deliver JSON payload               
    emit('my response', {'data': message['data']}) # emit() sends a message under a custom event name

@socketio.on('my broadcast event', namespace='/test')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True) # broadcast=True optional argument all clients connected to the namespace receive the message

                   
@socketio.on('connect', namespace='/test') 
def test_connect():                           #first argumenent is the event name, connect and disconnect are special event names the others are custom events
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


    # <!--  <div class="col-lg-6">
    #                      <img class="img-thumbnail panel panel-default" id="bgtwo" src="{{ url_for('video_feed_two') }}">
    #                 </div>
    #                 <div class="col-lg-6">
    #                      <img class="img-thumbnail panel panel-default" id="bgtwo" src="{{ url_for('video_feed_three') }}">
    #                 </div>
    #                  <div class="col-lg-6">
    #                      <img class="img-thumbnail panel panel-default" id="bgtwo" src="{{ url_for('video_feed_three') }}">
    #                 </div> -->


if __name__ == '__main__':
#    # app.run(host='0.0.0.0', debug=True)
     socketio.run(app, host='0.0.0.0', debug = True, use_reloader=False) #starts server on default port 5000 and makes socket connection available to other hosts (host = '0.0.0.0')
