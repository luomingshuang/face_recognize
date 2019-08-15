#-*- coding:utf-8 -*-
import cv2
import dlib
import numpy as np

#设置相应的参数
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

#设置存储路径
save_frames = "/home/lms/Documents/lipreading/face_recongnition/lipframes/"
save_file = "/home/lms/Documents/lipreading/face_recongnition/lipframesfile"

cap = cv2.VideoCapture(video)
while(True):
    n=1
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)   
    for face in faces:
        left = face.left()      #x
        top = face.top()        #y
        right = face.right()    #w
        w = right
        bottom = face.bottom()  #h
        face1 = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        shape = predictor(img, face)     #检测68个特征点
        cv2.imwrite(save_frames + 'lipframe%d.jpg'%n , face1) 
    with open(save_file, 'a') as file_object:
        file_object.write("lipframe%d.jpg\n"%n)
    n += 1 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
        shape = predictor(img, face)  # 寻找人脸的68个标定点 
        for index, pt in enumerate(shape.parts()):
            print('Part {}: {}'.format(index, pt))
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
            #cv::putText(temp,to_string(i), cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), CV_FONT_HERSHEY_PLAIN,1, cv::Scalar(0, 0, 255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(index+1), pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
             
        cv2.imshow("image", img)
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
        fr_height, fr_width = img.shape[:2]
        if x_ed > fr_width:
            x_ed = fr_width
            x_st = fr_width - width
        y_ed = int(min(fr_height, y_mouth_center + width/2))
        y_st = int(y_ed - width)
        if y_st < 0:
            y_st = 0
            y_ed = width
        mouth = img[y_st:y_ed, x_st:x_ed]
'''
        

'''
cap = cv2.imread(lms)
discern(cap, 1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''