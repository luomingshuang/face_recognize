#-*- coding:utf-8 -*-
import cv2
import dlib
import numpy as np

#设置参数
MIN_FACE_SIZE = 60
RECHECK_SIZE = 20
CR_RATE = 0.12
CR_SMALL_RATE = 0.04

detector = dlib.get_frontal_face_detector()  #使用默认的人脸分类器模型
predictor = dlib.shape_predictor("/home/lms/Py_SDK_20180627/build/shape_predictor_68_face_landmarks.dat")

#设置处理文件
video = "/home/lms/Documents/lipreading/face_recongnition/lipreading-croptime.mp4"
tupian = "/home/lms/Documents/lipreading/face_recongnition/aobama.jpg"
lms = "/home/lms/Documents/lipreading/face_recongnition/lmsface.jpg"
video1 = '/home/lms/server_50/share/DATASET/LRW/lipread_mp4/ABOUT/train/ABOUT_00001.mp4'

#设置存储路径
save_frames = "/home/lms/Documents/lipreading/face_recongnition/lipframes/"
save_file = "/home/lms/Documents/lipreading/face_recongnition/lipframesfile"

video_capture = cv2.VideoCapture(video1)
n=-1
while True:
 # Capture frame-by-frame
    n += 1
    ret, frame = video_capture.read()
    # print(ret)
    # print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)   
    for face in dets:
        left = face.left()      #x
        top = face.top()        #y
        right = face.right()    #w
        w = right
        bottom = face.bottom()  #h
        face1 = cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
        face1 = face1[top+4:bottom-1, left+4:right-1]
        shape = predictor(frame, face)
        for index, pt in enumerate(shape.parts()):
            #print('Part {}: {}'.format(index, pt))
            pt_pos = (pt.x, pt.y)
            cv2.circle(frame, pt_pos, 2, (255, 0, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX  #定义数字型号
            #cv2.putText(frame, str(index+1), pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        x_mouth_center = (shape.part(49).x + shape.part(55).x)/2 
        y_mouth_center = (shape.part(52).y + shape.part(58).y)/2
        mouth_center = np.array((x_mouth_center, y_mouth_center))      #嘴部中心坐标，生成嘴部坐标列表
        nose_center = np.array((shape.part(31).x, shape.part(31).y))   #鼻子中心坐标，生成鼻部坐标列表
        mn_dist = np.linalg.norm(mouth_center - nose_center)           #求范数，默认二范数，各数平方和的开方
        lr_dist = (1+CR_RATE)*shape.part(55).x - (1-CR_RATE)*shape.part(49).x
        if lr_dist > 0.75 * w:
            lr_dist = (1+CR_SMALL_RATE)*shape.part(55).x - (1-CR_SMALL_RATE)*shape.part(49).x
        width = int(max(2 * mn_dist, lr_dist))
        x_st = int(max(0, x_mouth_center - width/2))
        x_ed = int(x_st + width)
        fr_height, fr_width = face1.shape[:2]
        if x_ed > fr_width:
            x_ed = fr_width
            x_st = fr_width - width
        y_ed = int(min(fr_height, y_mouth_center + width/2))
        y_st = int(y_ed - width)
        if y_st < 0:
            y_st = 0
            y_ed = width
        mouth = face1[y_st:y_ed, x_st:x_ed]
        #cv2.imwrite(save_frames+'frame_%d.jpg'%n, mouth)
        #with open(save_file, 'a') as file_object:
        #    file_object.write("lipframe%d.jpg"%n + '\n') 
    cv2.imshow('Video', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break        
video_capture.release()
cv2.destroyAllWindows()