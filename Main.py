from threading import Thread
from TargetVision import Vision
from TargetTrack import Target

import time
import pygame
import sys
import math

'''
This is a demo for object absolute position and velocity tracking. 
It allows you to enter an (spherical) object's diameter, and use it to map its depth-distance
to the camera

'''


pygame.init()
width, height = 620, 480 # do not change, you will have to change this accross other files to make it work
dis = pygame.display.set_mode((width, height))
pygame.display.set_caption("camera-radar-view")

targetVision = Vision()
ball = Target(diameter="35")
ballX = 0
ballZ = 0


clock = pygame.time.Clock()
def update():
    clock.tick(60)

# define some pigame UI elements 
WHITE = (255, 255, 255)
font = pygame.font.Font(None, 36)
text = font.render("camera", True, WHITE)
text_rect = text.get_rect(center=(width // 2, 10))   
line_y = 30
line_start = (width // 2 - 50, line_y)
line_end = (width // 2 + 50, line_y)
line_length = 100
line1_start = (width // 2, line_y)
line1_end = (width // 2 + 50, line_y+50)
line2_start = (width // 2, line_y)
line2_end = (width // 2 - 50, line_y+50)


Thread(target=update, args=()).start()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
                targetVision.stopVision()


    targetVision.captureVision() # controls the cv2 camera capture and masking
    ballX, ballY, ballDiameter = targetVision.getBallInformation() # retrive pixel position information of contour
    targetVision.showVision() # display mask
    ball.updateStatus(ballX, ballY, ballDiameter, targetVision.getFps()) # feed pixel position information to Target object
    # calculate absolute X,Y,Z positions (no units yet)
    ball.setZ_pos() 
    ball.setX_pos()
    # use FPS to calculate velocity
    ball.setZ_velocity()
    ball.setX_velocity()
    print("Target's distance from camera: ",ball.getZ())
    
    # other functions to test V:
        # print(ball.getTargetDistance())
        # print(ball.getZVelocity())
        # ball.printTargetInformation()
        # ball.getBallTimeToCenter()
        # ball.getBall300To100Time()
        # print(ball.getX_velocity())

    # draw object absolute location on pyGame window
    ballX = ball.getX()
    ballZ = ball.getZ()
    dis.fill((0, 0, 0)) # disable this to see target path!!!
    pygame.draw.rect(dis, (255, 0, 0), pygame.Rect(310+ballX, ballZ/3, 30, 30))

    # draw background shapes
    dis.blit(text, text_rect)
    pygame.draw.line(dis, WHITE, line_start, line_end, 2)
    pygame.draw.line(dis, WHITE, line1_start, line1_end, 2)
    pygame.draw.line(dis, WHITE, line2_start, line2_end, 2)
    pygame.display.flip()

pygame.quit()