import cv2
import numpy as np
import imutils
import argparse
from FPS import FPS
from WebcamVideoStream import WebcamVideoStream


class Target:

    def __init__(self):
        self.xPixelPos = None
        self.yPixelPos = None
        self.xLastPixelPos = None
        self.yLastPixelPos = None
        self.xVelocity = 0
        self.xTempVelocity = self.xVelocity
        self.xAvgVelocity = 0
        self.yVelocity = 0
        self.yTempVelocity = self.yVelocity
        self.yAvgVelocity = 0
        self.pixelSize = 0
        self.lastPixelSize = 0
        self.xV_sumCount = 0
        self.yV_sumCount = 0
        self.xV_zeroCont = 0
        self.xVelocitySum = 0
        self.yVelocitySum = 0
        self.xVelocitySumNum = 5
        self.yVelocitySumNum = 5
        self.hasAverageVelocity = False;
        self.fps = 0


    def getBall_X_pixel_after(self, seconds):
        if self.hasAverageVelocity:
            return self.xPixelPos+(self.xAvgVelocity*seconds)
        else:
            return None

    #def getBall_X_pixel_after(self, seconds):

    def setBall_XY_Velocity(self):
        self.setX_pixelVelocityAverage()
        self.setY_pixelVelocityAverage()

    def setX_pixelVelocityAverage(self):
        if self.hasTarget():
            self.setX_pixelVelocity()
            if (self.xVelocity != 0):
                print(self.xVelocity, self.xV_sumCount)
                self.xVelocitySum += self.xVelocity
                self.xV_sumCount += 1
            else:
                self.xV_zeroCont += 1
                if self.xV_zeroCont >= 10:
                    self.xVelocitySum += 0
                    self.xV_sumCount += 1
                    self.xV_zeroCont = 0;

            if self.xV_sumCount == self.xVelocitySumNum:
                self.xV_sumCount = 0
                self.xAvgVelocity = self.xVelocitySum / self.xVelocitySumNum
                self.xVelocitySum = 0
                self.hasAverageVelocity = True;
                print("Xv Avg:", self.xAvgVelocity)

        else:
            self.xAvgVelocity = 0;
            self.hasAverageVelocity = False;

    def setX_pixelVelocity(self):
        if self.hasLastFrame():
            self.xVelocity = (self.xPixelPos-self.xLastPixelPos)/(1/self.fps)

        else:
            self.xVelocity = 0

    def setY_pixelVelocityAverage(self):
        if self.hasTarget():
            self.setY_pixelVelocity()
            if (self.yVelocity != 0):
                #print(self.yVelocity, self.yV_sumCount)
                self.yVelocitySum += self.yVelocity
                self.yV_sumCount += 1

            if self.yV_sumCount == self.yVelocitySumNum:
                self.yV_sumCount = 0
                self.yAvgVelocity = self.yVelocitySum / self.yVelocitySumNum
                self.yVelocitySum = 0
                self.hasAverageVelocity = True;
                #print("Yv Avg:", self.yAvgVelocity)

        else:
            self.xAvgVelocity = 0;
            self.hasAverageVelocity = False;

    def setY_pixelVelocity(self):
        if self.hasLastFrame():
            self.yVelocity = (self.yPixelPos - self.yLastPixelPos) / (1 / self.fps)
        else:
            self.yVelocity = 0

    def getYAvg(self):
        return self.yAvgVelocity

    def setX_PixelPos(self, xPixel):
        self.xPixelPos = xPixel

    def setY_PixelPos(self, yPixel):
        self.yPixelPos = yPixel

    def getX_PixelPos(self):
        return self.xPixelPos

    def getY_PixelPos(self):
        return self.yPixelPos



    def updateStatus(self, xPixelPos, yPixelPos, pixelSize, fps):
        self.xLastPixelPos = self.xPixelPos
        self.yLastPixelPos = self.yPixelPos
        self.xPixelPos = xPixelPos
        self.yPixelPos = yPixelPos
        self.lastPixelSize = self.pixelSize
        self.pixelSize = pixelSize
        self.fps = fps

    def hasLastFrame(self):
        return self.lastPixelSize and self.xLastPixelPos and self.yLastPixelPos != None

    def hasTarget(self):
        return self.pixelSize != 0

    def printTargetInformation(self):
        print("Ball: ","xPixPos:", self.xPixelPos,"yPixelPos:", self.yPixelPos,
              "xVelocity:", self.xVelocity,"yVelocity:", self.yVelocity,"pixelSize:", self.pixelSize,
              "xV:",self.xAvgVelocity,"yV",self.yAvgVelocity)

    def printBallPosition(self):
        print("Ball: ", "xPixPos:", self.xPixelPos, "yPixelPos:", self.yPixelPos)
    def printBallVelocity(self):
        print("xV:",self.xAvgVelocity,"yV",self.yAvgVelocity)

    def printBallRawVelocity(self):
        print("xV:", self.xVelocity, "yV", self.yVelocity, "xPos", self.xPixelPos, "yPos", self.yPixelPos)
