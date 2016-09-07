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
# from gevent import monkey
# monkey.patch_all()
#import redis
from flask import Flask, render_template, Response, redirect, url_for, request, jsonify, send_file
import Camera
from flask.ext.socketio import SocketIO,send, emit #Socketio depends on gevent
import SurveillanceSystem
import json
import logging
import threading
import time
from random import random



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

Home_Surveillance = SurveillanceSystem.Surveillance_System()


thread1 = threading.Thread() 
thread2 = threading.Thread() 
thread1.daemon = False
thread2.daemon = False




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
        frame = camera.read_jpg()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # builds 'packet' of data with header and payload 

@app.route('/video_feed_one')
def video_feed_one():

    return Response(gen(Home_Surveillance.cameras[0]),
                    mimetype='multipart/x-mixed-replace; boundary=frame') # a stream where each part replaces the previous part the multipart/x-mixed-replace content type must be used.

@app.route('/video_feed_two')
def video_feed_two():
    return Response(gen(Home_Surveillance.cameras[1]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/create_alert', methods = ['GET','POST'])
def create_alert():
    if request.method == 'POST':
        camera = request.form.get('camera')
        event = request.form.get('eventdetail')
        alarmstate = request.form.get('alarmstate')
        person = request.form.get('person')
        push_alert = request.form.get('push_alert')
        email_alert = request.form.get('email_alert')
        trigger_alarm = request.form.get('trigger_alarm')
        notify_police = request.form.get('notify_police')


        actions = {'push_alert': push_alert , 'email_alert':email_alert , 'trigger_alarm':trigger_alarm , 'notify_police':notify_police}
        with Home_Surveillance.alerts_lock:
            Home_Surveillance.alerts.append(SurveillanceSystem.Alert(alarmstate,camera, event, person, actions))  #alarmState,camera, event, person, action)
        Home_Surveillance.alerts[-1].id 
        data = {"alert_id": Home_Surveillance.alerts[-1].id, "alert_message": "Alert if " + Home_Surveillance.alerts[-1].alertString}
        return jsonify(data)
    return render_template('index.html')

@app.route('/remove_alert', methods = ['GET','POST'])
def remove_alert():
    if request.method == 'POST':
        alertID = request.form.get('alert_id')

        with Home_Surveillance.alerts_lock:
            for i, alert in enumerate(Home_Surveillance.alerts):
                if alert.id == alertID:
                    del Home_Surveillance.alerts[i]
                    break
           
        data = {"alert_status": "removed"}
        return jsonify(data)
    return render_template('index.html')

@app.route('/remove_face', methods = ['GET','POST'])
def remove_face():
    if request.method == 'POST':
        predicted_name = request.form.get('predicted_name')

        for camera in Home_Surveillance.cameras:
            with camera.people_dict_lock:
                try:    
                    del camera.people[predicted_name]    #removes face from people detected in all cameras - need to change this!!
                except:
                    pass

        data = {"face_removed":  'true'}
        return jsonify(data)
    return render_template('index.html')

@app.route('/add_face', methods = ['GET','POST'])
def add_face():
    if request.method == 'POST':
        new_name = request.form.get('new_name')
        predicted_name = request.form.get('predicted_name')

        for camera in Home_Surveillance.cameras:
            with camera.people_dict_lock:    
                img = camera.people[predicted_name].face   #gets face of person detected in cameras 
                del camera.people[predicted_name]    #removes face from people detected in all cameras - need to change this!!
        wriitenToDir = Home_Surveillance.add_face(new_name,img)
           
           
        data = {"face_added":  wriitenToDir}
        return jsonify(data)
    return render_template('index.html')

@app.route('/retrain_classifier', methods = ['GET','POST'])
def retrain_classifier():
    if request.method == 'POST':

        retrained = Home_Surveillance.trainClassifier()
                   
        data = {"finished":  retrained}
        return jsonify(data)
    return render_template('index.html')



@app.route('/get_faceimg/<name>')
def get_faceimg(name):
    #print "\n/////////////////////////////////////////////////image\n"+name+"\n/////////////////////////////////////////////////image\n"
    for camera in Home_Surveillance.cameras:    
            try:
                img = camera.people[name].thumbnail #need to change to get face from specific camera
            except:
                img = ""
                pass
              
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame') #send_file(img, mimetype='image/jpg')

def update_faces():
     #, 'imgurl': url_for('get_faceimg', name = key) 
     #print "\nsending face data/////////////////////////////////////////////////\n"
     while True:
            peopledata = []
            persondict = {}
            thumbnail = None
            for camera in Home_Surveillance.cameras:
                for key, obj in camera.people.iteritems():  
                   
                    persondict = {'identity': key , 'confidence': obj.confidence, 'image': '/get_faceimg/' + key}
                   
                    peopledata.append(persondict)
            #print json.dumps(peopledata)
     
            socketio.emit('people_detected', json.dumps(peopledata) ,namespace='/test')
            time.sleep(4)

def alarm_state():
     while True:
            alarmstatus = {'state': Home_Surveillance.alarmState , 'triggered': Home_Surveillance.alarmTriggerd }
            socketio.emit('alarm_status', json.dumps(alarmstatus) ,namespace='/test')
            time.sleep(2)


@socketio.on('alarm_state_change', namespace='/test') 
def alarm_state_change():   
    Home_Surveillance.change_alarmState()

@socketio.on('panic', namespace='/test') 
def panic(): 
    Home_Surveillance.trigger_alarm()
   

@socketio.on('my event', namespace='/test') #socketio used to receive websocket messages # Namespaces allow a cliet to open multiple connectiosn to the server that are multiplexed on a single socket
def test_message(message):   #custom events deliver JSON payload 

    emit('my response', {'data': message['data']}) # emit() sends a message under a custom event name

@socketio.on('my broadcast event', namespace='/test')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True) # broadcast=True optional argument all clients connected to the namespace receive the message

                   
@socketio.on('connect', namespace='/test') 
def test_connect():                           #first argumenent is the event name, connect and disconnect are special event names the others are custom events
    
    global thread2 #need visibility of global thread object
    global thread1
    print "\n\nclient connected\n\n"
    #Start the random number generator thread only if the thread has not been started before.
    # if not thread1.isAlive():
    #     print "Starting Thread1"
    #     thread1 = threading.Thread(name='websocket_process_thread_',target= random_number, args=())
    #     thread1.start()

    if not thread2.isAlive():
        print "Starting Thread2"
        thread2 = threading.Thread(name='websocket_process_thread_',target= update_faces, args=())
        thread2.start()

    if not thread1.isAlive():
        print "Starting Thread1"
        thread2 = threading.Thread(name='alarmstate_process_thread_',target= alarm_state, args=())
        thread2.start()

    #emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
#    # app.run(host='0.0.0.0', debug=True)
     socketio.run(app, host='0.0.0.0', debug = True, use_reloader=False) #starts server on default port 5000 and makes socket connection available to other hosts (host = '0.0.0.0')
    
