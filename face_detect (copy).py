#face—detect.py
#-*- coding : utf-8 -*-
import cv2
import sys

cascPath="E:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml"
shipin="E:\项目文件\唇语识别\lipreading-demo-20180629.mp4"
save_frame="E:/tupian/frames/"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(shipin)
n=0
while True:
 # Capture frame-by-frame
   n += 1
   ret, frame = video_capture.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor=1.1,
       minNeighbors=3,
       minSize=(30, 30),
       flags=cv2.CASCADE_SCALE_IMAGE
   )
   # Draw a rectangle around the faces
   faces_list=[] 
    
   for (x, y, w, h) in faces: 
      face = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      faces_list.append(face)

   faces_list = sorted(faces_list, key=lambda x: face.shape[0], reverse=True) 
  
   # Display the resulting frame
   face = faces_list[0]
   face = face[y:y+h, x:x+w]
   cv2.imshow('Video', frame)
   #cv2.imwrite(save_frame+'frame_%d.jpg'%n, face)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()