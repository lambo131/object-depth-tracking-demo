import cv2
import numpy as np
import imutils
import argparse
from FPS import FPS
from WebcamVideoStream import WebcamVideoStream



ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=1,
                help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

print("[INFO] sampling frames from webcam...")
stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
stream.set(3, 640)
stream.set(4, 480)
fps = FPS().start()
# loop over some frames
while fps._numFrames < args["num_frames"]:
    # grab the frame from the stream and resize it to have a maximum
    # width of 400 pixels
    (grabbed, frame) = stream.read()
    frame = imutils.resize(frame, width=400)
    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()



# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps2 = FPS().start()
# loop over some frames...this time using the threaded stream
while fps2._numFrames < args["num_frames"]:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    # update the FPS counter
    fps2.update()
# stop the timer and display FPS information
fps2.stop()
print("[INFO] elasped time: {:.2f}".format(fps2.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps2.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

