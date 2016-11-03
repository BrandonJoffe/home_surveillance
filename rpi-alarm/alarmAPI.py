# Raspberry PI GPIO Alarm interface
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

# This Flask implementation is used to demonstrate the potential of the
# system to be connected to an already existing alarm panel. In order
# to acheive this, slight changes to the code will be required, as well as
# the use of a few electrical components i.e transistor to create a open 
# and closed circuit. 

import RPi.GPIO as GPIO
import threading
import time
import json
from flask import Flask, render_template, request, Response,jsonify 

app = Flask(__name__)

GPIO.setmode(GPIO.BCM)
triggered = False

thread = threading.Thread()
thread.daemon = False

# Dictionary used to store the pin number, name, and pin state:
pins = {
   26 : {'name' : 'alarm', 'state' : GPIO.LOW},
   13 : {'name' : 'siren', 'state' : GPIO.LOW},
   19 : {'name' : 'active', 'state' : GPIO.HIGH}
   }

alarm_state = {'state': 0, 'triggered': triggered}
# Set each pin as an output and set it low:
for pin in pins:
   GPIO.setup(pin, GPIO.OUT)
   GPIO.output(pin, pins[pin]['state'])

@app.route("/", methods = ['GET','POST'])
def main():
   """Returns alarms current state"""
    global alarm_state
    global triggered
    if request.method == 'POST':
	    password = request.form.get('password')
	    if password == 'admin':		   
		    alarm_state['state'] = GPIO.input(13)
		    alarm_state['triggered'] = triggered
		    return jsonify(alarm_state)
	    else:
		    return 'Access Denied'
		   
@app.route("/change_state", methods = ['GET','POST'])
def change_state():
	"""Changes alarm's current state"""
    global triggered 
    global alarm_state
    if request.method == 'POST':
	    password = request.form.get('password')
	    if password == 'admin':	
	        GPIO.output(13, not GPIO.input(13))
		    if GPIO.input(13) == 0:
			    triggered = False			 	
		    alarm_state['state'] = GPIO.input(13)
		    alarm_state['triggered'] = triggered
		    return jsonify(alarm_state)
	    else:
		    return 'Access Denied'
		   
@app.route("/trigger", methods = ['GET','POST'])
def trigger():
   """Triggers the alarm"""
   global triggered
   if request.method == 'POST':
	   password = request.form.get('password')
	   if password == 'admin':
		   GPIO.output(26, GPIO.HIGH)
		   triggered = True	
		   global thread    
		   
		   if not thread.isAlive():
			   thread = threading.Thread(name='trigger_thread', target = alarmtrigger,args=())
			   thread.start()			   
		   alarm_state['state'] = GPIO.input(13)
		   alarm_state['triggered'] = triggered
		   
		   return jsonify(alarm_state)
	   else:
		   return 'Access Denied'

def alarmtrigger():
	global triggered
	while True:
		if triggered == True:	
			 GPIO.output(26, not GPIO.input(26))
			 time.sleep(0.25)
		else:
			GPIO.output(26, GPIO.LOW)
			time.sleep(1)
  
if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=False)
