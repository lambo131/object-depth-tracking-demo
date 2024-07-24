import cv2
import numpy as np
import imutils
import argparse
from FPS import FPS
from WebcamVideoStream import WebcamVideoStream

vs = WebcamVideoStream(src=0).start()
count = 0
framesPerSecond = 0
while True:
    if(count == 0):
        fps3 = FPS().start()

    img = vs.read()
    fps3.update()
    count+=1

    if(fps3._numFrames > 10):
        fps3.stop()
        framesPerSecond = fps3.fps()
        count = 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array(([20, 200, 150]))
    upper_red = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.erode(mask, None, iterations=5)
    mask2 = cv2.dilate(mask, None, iterations=6)
    result = cv2.bitwise_and(img, img, mask=mask)
    edged = cv2.Canny(mask2, 30, 200)


    cv2.imshow("mask", mask2)

    cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 5)
        # find the biggest countour (c) by the area
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(img, c, -1, (0, 0, 255), 5)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cX = int(x+(w/2))
        cY = int(y+(h/2))
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (cX-20, cY-20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(cX,cY)

    cv2.putText(img, str(framesPerSecond), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
vs.stop()