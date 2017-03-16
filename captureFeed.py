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
import sys

def main(argv):
    videofeed = ''
    videofeed = str(sys.argv)

    print "---------OpenCV Video feed analyzer -------------\n\n\n"
    print "Analyzing url: " + videofeed
    cap = cv2.VideoCapture(videofeed)

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

    while (cap.isOpened()):
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


if __name__ == "__main__":
   main(sys.argv[1:])