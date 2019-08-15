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
tupian = "/home/lms/Documents/lipreading/face_recongnition/aobama.jpg"
lms = "/home/lms/Documents/lipreading/face_recongnition/lmsface.jpg"
video1 = "/home/lms/LRW/video/ABOUT_00001.mp4"

#设置存储路径
save_frames = "/home/lms/Documents/lipreading/face_recongnition/lipframes/"
save_file = "/home/lms/Documents/lipreading/face_recongnition/lipframesfile"
save_path_info = "/home/lms/Documents/lipreading/face_recongnition/faceinfofiles"
save_path_asis = "/home/lms/Documents/lipreading/face_recongnition/lipcrop"
save_path_warp = "/home/lms/Documents/lipreading/face_recongnition/lipwrap"

detector = vipl.Detector(FD_PATH)       #人脸检测器
detector.set_size(20)                   #检测器设定尺寸
predictor = vipl.Predictor(PD_PATH)     #特征点检测器
regressor = vipl.PoseRegressor(PE_PATH) #姿态回归器


cap = cv2.VideoCapture(tupian)
n = -1
while True:
    n += 1
    info_path = '%s/%d.json' % (save_path_info, n)                     #信息路径
    ret, img = cap.read()
    # cv2.imshow(img)
    faces = detector(img)   
    for face in faces:
        x = int(face.x)
        y = int(face.y)
        w = int(face.w)
        h = int(face.h)
        #cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 0), 2)
        pts = predictor(img, x, y, w, h)
        yaw, pitch, roll = regressor(img, face)                        #yaw是偏航角,pitch是俯仰角,roll是翻滚角
        meta = {"x": x, "y": y, "w": w, "h": h, 
                "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
                "landmarks": [{'x': pt.x, 'y': pt.y} for pt in pts]}   #参数命名设定
        json.dump(meta, open(info_path, 'w'))

        #裁剪嘴巴区域
        y_center = (pts[48].y + pts[55].y) / 2  
        x_center = (pts[46].x + pts[47].x) / 2
        mouth_center = np.array((x_center, y_center))                  #嘴部中心坐标，生成嘴部坐标列表
        nose_center = np.array((pts[34].x, pts[34].y))                 #鼻子中心坐标，生成鼻部坐标列表
        mn_dist = np.linalg.norm(mouth_center - nose_center)           #求范数，默认二范数，各数平方和的开方
        lr_dist = (1+CR_RATE)*pts[47].x - (1-CR_RATE)*pts[46].x        #唇部的宽度距离
        if lr_dist > 0.75 * w:
            lr_dist = (1+CR_SMALL_RATE)*pts[47].x - (1-CR_SMALL_RATE)*pts[46].x
        width = int(max(2 * mn_dist, lr_dist))
        x_st = int(max(0, x_center - width/2))
        x_ed = int(x_st + width)
        fr_height, fr_width = img.shape[:2]
        if x_ed > fr_width:
            x_ed = fr_width
            x_st = fr_width - width
        y_ed = int(min(fr_height, y_center + width/2))
        y_st = int(y_ed - width)
        if y_st < 0:
            y_st = 0
            y_ed = width
        mouth = img[y_st:y_ed, x_st:x_ed, :]
        cv2.imwrite('%s/%d.jpg' % (save_path_asis, n), mouth)          #截取嘴部图片并保存下来

        #调整图片,使脸部水平
        eye_direction = (pts[9].x-pts[0].x, pts[9].y-pts[0].y)          #眼睛的距离坐标
        rot = np.arctan2(eye_direction[1], eye_direction[0])/math.pi*180#rot:转向率,计算图像的旋转角

        rot = (rot - roll)/2                                            #综合眼睛和头部的旋转角度折中取均值
        image_center = (fr_width / 2.0, fr_height / 2.0)
        rot_mat = cv2.getRotationMatrix2D(image_center, rot, 1.0)       #绕某点旋转函数,返回一个旋转矩阵
        #调整图片边界,防止坐标溢出
        abs_cos = abs(rot_mat[0,0]) 
        abs_sin = abs(rot_mat[0,1])
        bound_w = int(fr_height * abs_sin + fr_width * abs_cos)
        bound_h = int(fr_height * abs_cos + fr_width * abs_sin)
        rot_mat[0, 2] += bound_w/2 - image_center[0]
        rot_mat[1, 2] += bound_h/2 - image_center[1]

        pts_warped = [{} for _ in range(81)]
        for i in range(0, 81):
            pts_warped[i]['x'], pts_warped[i]['y'] = np.dot(rot_mat, np.array([pts[i].x, pts[i].y, 1]))
                                                                        #每个点进行仿射变换
        warped = cv2.warpAffine(img, rot_mat, (bound_w, bound_h), flags=cv2.INTER_CUBIC)
                                                                        #warped表示img仿射变换后的图片
        #后面的过程步骤同上,修正剪切区域
        y_center_warped = (pts_warped[48]['y'] + pts_warped[55]['y']) / 2
        x_center_warped = (pts_warped[46]['x'] + pts_warped[47]['x']) / 2
        mouth_center_warped = np.array((x_center_warped, y_center_warped))
        nose_center_warped = np.array((pts_warped[34]['x'], pts_warped[34]['y']))
        mn_dist_warped = np.linalg.norm(mouth_center_warped - nose_center_warped)
        lr_dist_warped = (1+CR_RATE)*pts_warped[47]['x'] - (1-CR_RATE)*pts_warped[46]['x']
        if lr_dist_warped > 0.75 * w:
            lr_dist_warped = (1+CR_SMALL_RATE)*pts_warped[47]['x'] - (1-CR_SMALL_RATE)*pts_warped[46]['x']
        width_warped = int(max(2 * mn_dist, lr_dist_warped))
        x_st_warped = int(max(0, x_center_warped - width_warped/2))
        x_ed_warped = int(x_st_warped + width_warped)
        if x_ed_warped > bound_w:
            x_ed = bound_w
            x_st = bound_w - width_warped
        y_ed_warped = int(min(bound_h, y_center_warped + width_warped/2))
        y_st_warped = int(y_ed_warped - width_warped)
        if y_st_warped < 0:
            y_st_warped = 0
            y_ed_warped = width_warped
        mouth_warped = warped[y_st_warped:y_ed_warped, x_st_warped:x_ed_warped, :]
        cv2.imwrite('%s/%d.jpg' % (save_path_warp, n), mouth_warped)    #保存包装修剪后的嘴部图片

        with open(save_file, 'a') as file_object:
            file_object.write("lipframe%d.jpg"%n + '\n') 
    cv2.imshow('Video', img) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break        
    
cap.release()
cv2.destroyAllWindows()