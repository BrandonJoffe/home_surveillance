# main.py

from flask import Flask, render_template, Response
import Camera
from flask.ext.socketio import SocketIO, emit #Socketio depends on gevent
import SurveillanceSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

Home_Surveillance = SurveillanceSystem.Surveillance_System()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Home_Surveillance.cameras[0]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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



if __name__ == '__main__':
   # app.run(host='0.0.0.0', debug=True)
    socketio.run(app, host='0.0.0.0') #starts server on default port 5000
