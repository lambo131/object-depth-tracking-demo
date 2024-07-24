import datetime
import math
import time

class Target:

    def __init__(self, **kwargs):
        self.dectionDistance = 4
        self.cameraWidthToDistanceRation = 1.2
        self.screenWidthX = 620;
        self.screenHeightY = 480;


        self.xPixelPos = None
        self.xPos = None
        self.xStartTrackPos = 0
        self.xVelocity = 0
        self.xStartTrack = False
        self.xTravelTime = 0;
        self.xStartTrackTime = 0

        self.yPixelPos = None
        self.yPos = None
        self.yStartTrackPos = 0
        self.yVelocity = 0
        self.yStartTrack = False
        self.yTravelTime = 0;
        self.yStartTrackTime = 0

        self.zValue = -1 #mm
        self.zVelocity = 0
        self.zAvgVelocity = 0
        self.zStartTrackValue = 0
        self.zTravelTime = 0;
        self.zStartTrack = False
        self.zStartTrackTime = 0
        self.zDetectionDistance = 20
        self.xProjectionLength = 0
        self.yProjectionLength = 0

        self.ballPixelDiameter = 0

        for key, value in kwargs.items():
            if key == "diameter":
                self.targetDiameter = int(value)  # mm


        self.fps = 0

        self.state = -1
        self.timer = 0
        self.timeUsed = 0
        self.timeToCenter = 0

    def getBall300To100Time(self):
        timeNow = time.thread_time()

        if (self.state == -1):

            if self.zValue != -1:
                if self.zValue < 300:
                    self.timer = timeNow
                    if (self.zVelocity != 0):
                        self.timeToCenter = (-200) / self.zVelocity
                        self.state = 0

        if self.state == 0:
            if self.zValue < 100:
                self.timeUsed = timeNow - self.timer
                self.state = 1

        if self.state == 1:
            if self.hasTarget() == False:
                self.state = 2
        if self.state == 2:
            if self.zValue > 300:
                self.state = -1
        if self.timeUsed != None:
            print("Time used:", self.timeUsed, end=" | ")
        print("Predicted time:", self.timeToCenter, end=" | ")
        print("timer:", self.timer, end=" | ")
        print("ball:", self.ballPixelDiameter, end=" | ")
        print("state:", self.state)


    def getBallTimeToCenter(self):
        timeNow = time.thread_time()

        if (self.state == -1):

            if self.hasTarget():
                if self.xPos > -200 and self.xPos < 200:
                    self.timer = timeNow
                    if (self.xVelocity != 0):
                        self.timeToCenter = (-self.xPos) / self.xVelocity
                    if self.xPos < 0:
                        self.state = 0
                    else:
                        self.state = 1

        if self.state == 0:
            if self.xPos > 0:
                self.timeUsed = timeNow - self.timer
                self.state = 2
        if self.state == 1:
            if self.xPos < 0:
                self.timeUsed = timeNow - self.timer
                self.state = 2
        if self.state == 2:
            if self.hasTarget() == False:
                self.state = -1
        if self.timeUsed != None:
            print("Time used:", self.timeUsed, end=" | ")
        print("Predicted time:", self.timeToCenter, end=" | ")
        print("timer:", self.timer, end=" | ")
        print("ball:", self.targetDiameter, end=" | ")
        print("state:", self.state)


    def setZ_velocity(self):
        if self.hasTarget():
            if self.zStartTrack == False:
                self.zStartTrackValue = self.zValue
                self.zStartTrackTime = datetime.datetime.now()
                self.zStartTrack = True
            if self.zStartTrack == True:

                timeNow = datetime.datetime.now()
                totalTime = (timeNow - self.zStartTrackTime).total_seconds()
                if(totalTime > 0.3):
                    self.zVelocity = 0
                    self.zStartTrack = False
                if abs(self.zValue - self.zStartTrackValue) > self.zDetectionDistance:
                    if totalTime != 0:
                        self.zVelocity = (self.zValue - self.zStartTrackValue)/totalTime
                        self.zStartTrack = False
            return self.zVelocity
        else:
            pass

    def setZ_pos(self):
        if self.ballPixelDiameter != 0:
            self.xProjectionLength = (640 / self.ballPixelDiameter) * self.targetDiameter
            self.yProjectionLength = (self.screenHeightY / self.ballPixelDiameter) * self.targetDiameter
            self.zValue = (self.xProjectionLength) / self.cameraWidthToDistanceRation
        else:
            self.zValue = -1

    def setX_pixelVelocity(self): #don't use
        if self.hasTarget():
            if self.xStartTrack == False:
                self.xStartTrackPos = self.xPixelPos
                self.xStartTrackTime = datetime.datetime.now()
                self.xStartTrack = True
            if self.xStartTrack == True:

                timeNow = datetime.datetime.now()
                totalTime = (timeNow - self.xStartTrackTime).total_seconds()
                if(totalTime > 0.3):
                    self.xPixelVelocity = 0
                    self.xStartTrack = False
                if abs(self.xPixelPos - self.xStartTrackPos) > self.dectionDistance:
                    self.xPixelVelocity = (self.xPixelPos - self.xStartTrackPos)/totalTime
                    self.xStartTrack = False
        else:
            pass

    def setX_velocity(self):
        if self.hasTarget():
            if self.xStartTrack == False:
                self.xStartTrackPos = self.xPos
                self.xStartTrackTime = datetime.datetime.now()
                self.xStartTrack = True
            if self.xStartTrack == True:

                timeNow = datetime.datetime.now()
                totalTime = (timeNow - self.xStartTrackTime).total_seconds()
                if(totalTime > 0.3):
                    self.xVelocity = 0
                    self.xStartTrack = False
                if abs(self.xPos - self.xStartTrackPos) > self.dectionDistance:
                    if totalTime != 0:
                        self.xVelocity = (self.xPos - self.xStartTrackPos)/totalTime
                        self.xStartTrack = False
        else:
            pass

    def setX_pos(self):
        self.xPos = -((self.xPixelPos-self.screenWidthX/2)/self.screenWidthX)*self.xProjectionLength

    def setY_velocity(self):
        if self.hasTarget():
            if self.yStartTrack == False:
                self.yStartTrackPos = self.yPos
                self.yStartTrackTime = datetime.datetime.now()
                self.yStartTrack = True
            if self.yStartTrack == True:

                timeNow = datetime.datetime.now()
                totalTime = (timeNow - self.yStartTrackTime).total_seconds()
                if(totalTime > 0.3):
                    self.yVelocity = 0
                    self.yStartTrack = False
                if abs(self.yPos - self.yStartTrackPos) > self.dectionDistance:
                    if totalTime != 0:
                        self.yVelocity = (self.yPos - self.yStartTrackPos)/totalTime
                        self.yStartTrack = False
        else:
            pass

    def setY_pos(self):
        self.yPos = -((self.yPixelPos-self.screenHeightY/2)/self.screenHeightY)*self.yProjectionLength

    def setXYZdata(self):
        self.setZ_pos()
        self.setZ_velocity()
        self.setX_pixelVelocity()
        self.setX_velocity()
        self.setX_pos()
##################################
    def getZ(self):
        return self.zValue

    def getX(self):
        return self.xPos

    def getY(self):
        return self.yPos

    def getZ_velocity(self):
        return self.zVelocity

    def getX_velocity(self):
        return self.xVelocity

    def getY_velocity(self):
        return self.yVelocity

    def getX_pixelVelocity(self):
        return self.xPixelVelocity

    def getX_pixelPos(self):
        return self.xPixelPos


    def updateStatus(self, xPixelPos, yPixelPos, ballPixelDiameter, fps):
        self.xPixelPos = xPixelPos
        self.yPixelPos = yPixelPos
        self.ballPixelDiameter = ballPixelDiameter
        self.fps = fps


    def hasTarget(self):
        return self.ballPixelDiameter != -1

    def printTargetInformation(self):
        print("Ball: ","xPixPos:", self.xPixelPos,"yPixelPos:", self.yPixelPos,
              "xPixelVelocity:", self.xPixelVelocity,"yPixelVelocity:", self.yPixelVelocity,"pixelSize:", self.pixelSize,
              "xV:",self.xAvgVelocity,"yV",self.yAvgVelocity)

    def printBallPosition(self):
        print("Ball: ", "xPixPos:", self.xPixelPos, "yPixelPos:", self.yPixelPos)
    def printBallVelocity(self):
        print("xV:",self.xAvgVelocity,"yV",self.yAvgVelocity)

    def printBallRawVelocity(self):
        print("xV:", self.xVelocity, "yV", self.yVelocity, "xPos", self.xPixelPos, "yPos", self.yPixelPos)
