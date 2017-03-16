#!/usr/bin/python
#
#
# side tool to capture a feed with openCV and get an analysis of it
# Mathieu Duperre
# 2017
#
#

import numpy as np
import cv2
import cv2.cv as cv
import sys
import time
import os

def main(argv):
    videofeed = ''
    videofeed = str(sys.argv[1])

    print "\n\n-----------FFMEG video data output:-----------\n\n"
    os.system("ffmpeg -i " + videofeed)

    print "---------OpenCV Video feed analyzer -------------\n\n\n"
    print "Analyzing url: " + videofeed
    cap = cv2.VideoCapture(videofeed)
    cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') )

    if not cap.isOpened():
        try:
            cap.open()
        except ValueError:
            print "Could not open the video feed."
            exit()

    print "Video feed open."
    dump_video_info(cap)  # logging every specs of the video feed

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

    #recording a video sample (output.avi) for X seconds
    t_end = time.time() + 15
    while time.time() < t_end:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)

            # write the flipped frame
            out.write(frame)

            #        cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def dump_video_info(cap):
    print "---------Dumping video feed info---------------------"
    print "Position of the video file in milliseconds or video capture timestamp: "
    print cap.get(cv.CV_CAP_PROP_POS_MSEC)
    print "0-based index of the frame to be decoded/captured next: "
    print cap.get(cv.CV_CAP_PROP_POS_FRAMES)
    print "Relative position of the video file: 0 - start of the film, 1 - end of the film: "
    print cap.get(cv.CV_CAP_PROP_POS_AVI_RATIO)
    print "Width of the frames in the video stream: "
    print cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    print "Height of the frames in the video stream: "
    print cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
    print "Frame rate:"
    print cap.get(cv.CV_CAP_PROP_FPS)
    print "4-character code of codec."
    print cap.get(cv.CV_CAP_PROP_FOURCC)
    print "Number of frames in the video file."
    print cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
    print "Format of the Mat objects returned by retrieve() ."
    print cap.get(cv.CV_CAP_PROP_FORMAT)
    print "Backend-specific value indicating the current capture mode."
    print cap.get(cv.CV_CAP_PROP_MODE)
    print "Brightness of the image (only for cameras)."
    print cap.get(cv.CV_CAP_PROP_BRIGHTNESS)
    print "Contrast of the image (only for cameras)."
    print cap.get(cv.CV_CAP_PROP_CONTRAST)
    print "Saturation of the image (only for cameras)."
    print cap.get(cv.CV_CAP_PROP_SATURATION)
    print "Hue of the image (only for cameras)."
    print cap.get(cv.CV_CAP_PROP_HUE)
    print "Gain of the image (only for cameras)."
    print cap.get(cv.CV_CAP_PROP_GAIN)
    print "Exposure (only for cameras)."
    print cap.get(cv.CV_CAP_PROP_EXPOSURE)
    print "Boolean flags indicating whether images should be converted to RGB."
    print cap.get(cv.CV_CAP_PROP_CONVERT_RGB)
    print "--------------------------End of video feed info---------------------"

if __name__ == "__main__":
   main(sys.argv[1:])