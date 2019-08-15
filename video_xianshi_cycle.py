#encoding:utf-8
import cv2 
import numpy as np
video = '/home/lms/LRW/video/ABOUT_00001.mp4'
picture = "/home/lms/Documents/lipreading/face_recongnition/aobama.jpg" 
cap = cv2.VideoCapture(video) 
while(1):
     # get a frame 
    ret, frame = cap.read() 
    # show a frame 
    cv2.imshow("capture", frame) 
    if cv2.waitKey(100) & 0xFF == ord('q'): 
        break 
cap.release() 
cv2.destroyAllWindows() 


# img = cv2.imread(picture, -1)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 