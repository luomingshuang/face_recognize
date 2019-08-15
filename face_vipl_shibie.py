#-*- coding:utf-8 -*-
#注意:运行代码指令:CUDA_VISIBLE_DEVICES="" python3 face_video.py
#导入相应的模块
import cv2    
import numpy as np
import math
import sys, os, time
import json, pprint
import vipl

#设置相应的参数
MIN_FACE_SIZE = 60
RECHECK_SIZE = 20
CR_RATE = 0.12
CR_SMALL_RATE = 0.04

#设置检测器
FD_PATH = "/home/lms/Py_SDK_20180627/models/VIPLFaceDetector5.1.0.dat"                 #人脸检测器
PD_PATH = "/home/lms/Py_SDK_20180627/models/VIPLPointDetector5.0.pts81.stable.dat"     #特征点检测器
PE_PATH = "/home/lms/Py_SDK_20180627/models/VIPLPoseEstimation1.1.0.ext.dat"           #姿态检测器，在不同的头部姿态下检测出人脸

#设置处理文件（视频和图片）
video = "/home/lms/Documents/lipreading/face_recongnition/lipreading-croptime.mp4"
tupian1 = "/home/lms/Documents/lipreading/face_recongnition/people1.jpg"
tupian2 = "/home/lms/Documents/lipreading/face_recongnition/peichun.jpg"
lms = "/home/lms/Documents/lipreading/face_recongnition/lmsface.jpg"
lip1 = "/home/lms/LRW/crop_img/ABOUT/train/ABOUT_00001/1.jpg"

#设置存储路径
save_frames = "/home/lms/Documents/lipreading/face_recongnition/lipframes/"
save_file = "/home/lms/Documents/lipreading/face_recongnition/lipframesfile"
save_path_info = "/home/lms/Documents/lipreading/face_recongnition/faceinfofiles"
save_path_asis = "/home/lms/Documents/lipreading/face_recongnition/lipcrop"
save_path_warp = "/home/lms/Documents/lipreading/face_recongnition/lipwrap"

detector = vipl.Detector(FD_PATH)       #人脸检测器
detector.set_size(40)                   #检测器设定尺寸
predictor = vipl.Predictor(PD_PATH)     #特征点检测器
regressor = vipl.PoseRegressor(PE_PATH) #姿态回归器


img = cv2.imread(tupian2)
faces = detector(img)   
for face in faces:
    x = int(face.x)
    y = int(face.y)
    w = int(face.w)
    h = int(face.h)
    cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 0), 2)
print(x)
pts = predictor(img, x, y, w, h)
for index, pt in enumerate(pts):
    pos = (int(pt.x), int(pt.y))
    cv2.circle(img, pos, 3, color=(0, 0, 255))
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, str(index+1), pos, font, 0.3, (0,0,255), 1, cv2.LINE_AA)
    

        
cv2.imshow('Video', img) 
cv2.imwrite("shencheng.jpg", img)
cv2.waitKey(0)
#cv2.destroyAllWindows()