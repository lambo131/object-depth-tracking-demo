from threading import Thread

import cv2
import numpy as np
import imutils
from WebcamVideoStream import WebcamVideoStream
from FPS import FPS

class Vision:


    def __init__(self):
        self.vs = WebcamVideoStream(src=0)
        self.vs.start()
        self.fpsFinder = FPS()
        self.img = self.vs.read()
        self.mask = self.img
        self.count = 0
        self.framesPerSecond = -1
        self.showFps = True
        self.showBallContour = True
        self.showMask = True
        self.stopped = False
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def captureVision(self):
        #ret, self.img = self.cap.read()

        if (self.count == 0):
            self.fpsFinder = FPS().start()

        self.img = self.vs.read()
        self.fpsFinder.update()
        self.count += 1

        if (self.fpsFinder._numFrames > 30):
            self.fpsFinder.stop()
            self.framesPerSecond = self.fpsFinder.fps()
            self.count = 0


    def showFPS(self, toogle):
        if(toogle == True):
            self.showFps = True
        else:
            self.showFps = False
    def getFps(self):
        return self.framesPerSecond

    def stopVision(self):
        cv2.destroyAllWindows()
        self.vs.stop()
        self.stopped = True

    def showVision(self):
        #thread(target=self.update, args=()).start()
        percentage = 100
        width = int(self.img.shape[1] * percentage / 100)
        height = int(self.img.shape[0] * percentage / 100)
        dim = (width, height)
        # img2 = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA)
        # mask2 = cv2.resize(self.mask, dim, interpolation=cv2.INTER_AREA)
        if (self.showFps == True):
            cv2.putText(self.img, str(self.framesPerSecond), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
        if self.stopped == False:
            cv2.imshow("frame", self.img)
            if (self.showMask == True):
                pass
                cv2.imshow("mask", self.mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stopVision()

    def showBallContour(self, state):
        self.showBallContour = state

    def getBallSize(self):
        return self.ballPixelDiameter

    def isThereTarget(self):
        return self.ballSize != 0


    def getBallInformation(self):
        blur = cv2.GaussianBlur(self.img, (11, 11), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array(([21, 150, 50]))
        upper_yellow = np.array([40, 255, 255])
        lower_blue = np.array([110, 150, 150])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        self.mask = mask
        cnts = cv2.findContours(mask, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        #'''
        if len(cnts) > 0:
            # find the biggest countour (c) by the area
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cX = int(x+(w/2))
            cY = int(y+(h/2))
            if self.showBallContour == True:
                pass
                #cv2.drawContours(self.img, c, -1, (0, 0, 255), 5)
                #cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if w>h:
                ballPixelDiameter = w
            else:
                ballPixelDiameter = h
            return cX,cY, ballPixelDiameter

        self.ballPixelDiameter = 0

        return -1,-1,-1

