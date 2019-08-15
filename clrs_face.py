#-*- coding:utf-8 -*-
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
#SAVE_ROOT='test'

#申明相应的要引用的文件的路径
SAVE_ROOT = '/home/zhineng/server/40/DATASET/PART_TV_DATA/aft_final_check/add_path_lip'
FD_PATH = "/home/lms/Py_SDK_20180627/models/VIPLFaceDetector5.1.0.dat"                 #人脸检测器
PD_PATH = "/home/lms/Py_SDK_20180627/models/VIPLPointDetector5.0.pts81.stable.dat"     #特征点检测器
PE_PATH = "/home/lms/Py_SDK_20180627/models/VIPLPoseEstimation1.1.0.ext.dat"           #姿态检测器，在不同的头部姿态下检测出人脸


def face_size(info):                  #定义面部尺寸大小的函数
    return np.sqrt(info.w * info.h)   #sqrt(x)表示返回x的平方根


# Print iterations progress (from StackExchange)
#信息打印函数
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    #print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


#获取视频帧函数
def get_video_frames(name):
    imgs = []
    squeeze_flag = False
    extend_flag = False
    
    if os.path.exists('/'.join(name.split('/')[:-1]) + '/squeeze'):  #判断文件或者文件夹是否存在，os.path.exists(path) 
                                                                    #如果path存在返回True如果Path不存在返回False
        squeeze_flag = True
    elif os.path.exists('/'.join(name.split('/')[:-1]) + '/extend'):
        extend_flag = True
    ignore_list = '/'.join(name.split('/')[:-1]) + '/ignore'
    ignored = False
    if os.path.exists(ignore_list):
        ignores = [x.rstrip() for x in open(ignore_list, 'r').readlines()]#从图片目录名中读取每一行
        ignored = name.split('/')[-1] in ignores
    if not ignored and (squeeze_flag or extend_flag):
        if squeeze_flag:
            for file in sorted(os.listdir(name), key=lambda x: int(x.split('_')[0])):
                try:
                    img = cv2.imread('%s/%s' % (name, file))
                    height, width = img.shape[:2]
                    size = (int(width*3/4), height)
                    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
                    imgs.append(img)
                except Exception:
                    print ('Corrupt or missing file:', file)
                    input()
                    pass
        elif extend_flag:
            for file in sorted(os.listdir(name), key=lambda x: int(x.split('_')[0])):
                try:
                    img = cv2.imread('%s/%s' % (name, file))
                    height, width = img.shape[:2]
                    size = (int(width*64/45), height)
                    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
                    imgs.append(img)
                except Exception:
                    print ('Corrupt or missing file:', file)
                    input()
                    pass
    else:
        for file in sorted(os.listdir(name), key=lambda x: int(x.split('_')[0])):
            try:
                img = cv2.imread('%s/%s' % (name, file))   #将图片按顺序加到一个数组里
                imgs.append(img)
            except Exception:
                print ('Corrupt or missing file:', file)
                input()
                pass
    return imgs   #本函数返回一个图片序列数组

#运行主函数
if __name__ == "__main__":
    detector = vipl.Detector(FD_PATH)   #人脸检测器
    detector.set_size(MIN_FACE_SIZE)    #检测器设定尺寸
    predictor = vipl.Predictor(PD_PATH)   #预测器
    regressor = vipl.PoseRegressor(PE_PATH)     #回归器

    if sys.argv[1].endswith('.txt'):   #从外界获取的除了源代码文件外的第一个参数（不一定是数字），此处判断这个参数是不是以.txt结尾的
        # sys.argv[1] -- file list
        # sys.argv[2] -- base path for these files ending with '/'
        file_list = open(sys.argv[1], 'r').readlines()  #打开文件，并读取他的每一行
        if file_list[0].rstrip().endswith('.mp4'):      #rstrip()删除末尾的空格
            # Linux find specification
            file_list = [file.rstrip()[:-4] for file in file_list]
        names = [sys.argv[2] + name.split(' ')[0] for name in file_list]  #以空格分隔开之后，取第一个元素
    else:
        names = []
        # sys.argv[1] -- folder containing video files
        for root, dirs, files in os.walk(sys.argv[1]):  #os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
            for name in files:
                if name.endswith('.mp4'):
                    #print (name)
                    names.append(os.path.join(root, name)[:-4])
    total = len(names)
    glob_cnt = 0
    print ("---> %d files to process" % total)
    for name in names:
        glob_cnt += 1
        sub_path = '/'.join(name.split('/')[-4:])
        save_path_info = SAVE_ROOT + '/info/%s' % sub_path
        noface_path = "%s/no_face.txt" % save_path_info
        if os.path.exists(noface_path):
            continue
        save_path_asis = SAVE_ROOT + '/chop/%s' % sub_path
        save_path_warp = SAVE_ROOT + '/crop/%s' % sub_path
        prev_chop_path = SAVE_ROOT + '/preview_chop/%s' % sub_path
        prev_crop_path = SAVE_ROOT + '/preview_crop/%s' % sub_path
        for target in [save_path_info, save_path_asis, save_path_warp]:#, save_path_asis+'/bad', save_path_warp+'/bad']:
            os.makedirs(target, exist_ok = True)   #创建路径
        frames = get_video_frames(name)  #返回视频帧的序列图片
        seq_l = len(frames)  #帧的数目
        cnt = -1
        printProgressBar(0, seq_l, prefix = '[%d/%d] %s:' % (glob_cnt, total, sub_path), suffix = '', length = 50)
        for frame in frames:
            cnt += 1
            info_path = '%s/%d.json' % (save_path_info, cnt)  #信息路径
            angle_path = save_path_info + '/angle'
            start_t = time.time()
            if os.path.exists(angle_path):
                largest_angle = int(open(angle_path, 'r').read())
            else:
                largest_angle = -1
            if not os.path.exists(info_path):   
                fr_height, fr_width = frame.shape[:2]  #返回每个帧的高度和宽度，两个尺寸
                faces = sorted(detector(frame), key=lambda x: face_size(x), reverse=True)   #检测器检测每帧图片中的人脸，设定人脸区域的尺寸，一张图片上多个人脸
                if len(faces) == 0:                       #如果没有检测到人脸图片，重新设定检测区域尺寸      #并按照检测出的人脸的区域的尺寸大小从大到小降序
                    detector.set_size(RECHECK_SIZE)
                    faces = sorted(detector(frame), key=lambda x: face_size(x), reverse=True)
                detector.set_size(MIN_FACE_SIZE)
                fail_flag = False if len(faces) > 0 else True 
                if fail_flag:
                    # Failed to detect face on frame
                    continue
                
                face = faces[0]  #取图片中最大的人脸区域
                x, y, w, h = int(face.x), int(face.y), int(face.w), int(face.h)   #取这个人脸区域的起始坐标，宽度和高度
                pts = predictor(frame, x, y, w, h)
                yaw, pitch, roll = regressor(frame, face)
                #bad_flag = False # TODO

                meta = {"x": x, "y": y, "w": w, "h": h, 
                        "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
                        "landmarks": [{'x': pt.x, 'y': pt.y} for pt in pts]}   #参数命名设定
                frame_angle = int(abs(pitch * 0.3) + abs(roll * 0.3) + abs(yaw * 0.4))   
                if frame_angle > largest_angle:
                    largest_angle = frame_angle
                    open(angle_path, 'w').write(str(largest_angle))   #写入
                json.dump(meta, open(info_path, 'w'))

                # Crop the mouth region directly.  #剪辑嘴巴区域
                y_center = (pts[48].y + pts[55].y) / 2  
                x_center = (pts[46].x + pts[47].x) / 2
                mouth_center = np.array((x_center, y_center))   #嘴部中心坐标，生成嘴部坐标列表
                nose_center = np.array((pts[34].x, pts[34].y))   #鼻子中心坐标，生成鼻部坐标列表
                mn_dist = np.linalg.norm(mouth_center - nose_center)  #求范数，默认二范数，各数平方和的开方
                lr_dist = (1+CR_RATE)*pts[47].x - (1-CR_RATE)*pts[46].x
                if lr_dist > 0.75 * w:
                    lr_dist = (1+CR_SMALL_RATE)*pts[47].x - (1-CR_SMALL_RATE)*pts[46].x
                width = int(max(2 * mn_dist, lr_dist))
                x_st = int(max(0, x_center - width/2))
                x_ed = int(x_st + width)
                if x_ed > fr_width:
                    x_ed = fr_width
                    x_st = fr_width - width
                y_ed = int(min(fr_height, y_center + width/2))
                y_st = int(y_ed - width)
                if y_st < 0:
                    y_st = 0
                    y_ed = width
                mouth = frame[y_st:y_ed, x_st:x_ed, :]
                cv2.imwrite('%s/%d.jpg' % (save_path_asis, cnt), mouth)   #截取嘴部图片，并将检测到的嘴部图片保存下来，写入文件夹
                #cv2.imwrite('%s/%s%d.jpg' % (save_path_asis, 'bad/' if bad_flag else '', cnt), mouth)
                '''
                # Draw the chopped region on the original frame.
                chop = frame.copy()
                for pt in pts:
                    cv2.circle(chop, (int(pt.x), int(pt.y)), 4, (0, 255, 0), -1)
                cv2.rectangle(chop, (x, y), (x+w, y+h), (255, 0, 0), 2, 8, 0)
                cv2.circle(chop, (int(x_center), int(y_center)), 4, (255, 0, 0), -1)
                cv2.circle(chop, (int(pts[34].x), int(pts[34].y)), 4, (255, 0, 0), -1)
                cv2.rectangle(chop, (x_st, y_st), (x_ed, y_ed), (0, 0, 255), 3, 8, 0)
                os.makedirs(prev_chop_path, exist_ok = True)
                cv2.imwrite('%s/%d.jpg' % (prev_chop_path, cnt), chop)
                '''
                # Rotate the image so that the face is level.   #调整图片
                eye_direction = (pts[9].x-pts[0].x, pts[9].y-pts[0].y)
                rot = np.arctan2(eye_direction[1], eye_direction[0])/math.pi*180
                # NOTE: using eyes only is not enough for a smooth crop
                rot = (rot - roll)/2
                image_center = (fr_width / 2.0, fr_height / 2.0)
                rot_mat = cv2.getRotationMatrix2D(image_center, rot, 1.0)
                # Find the new width and height bounds to avoid coordinate overflow.
                abs_cos = abs(rot_mat[0,0]) 
                abs_sin = abs(rot_mat[0,1])
                bound_w = int(fr_height * abs_sin + fr_width * abs_cos)
                bound_h = int(fr_height * abs_cos + fr_width * abs_sin)
                rot_mat[0, 2] += bound_w/2 - image_center[0]
                rot_mat[1, 2] += bound_h/2 - image_center[1]

                pts_warped = [{} for _ in range(81)]
                for i in range(0, 81):
                    pts_warped[i]['x'], pts_warped[i]['y'] = np.dot(rot_mat, np.array([pts[i].x, pts[i].y, 1]))
                warped = cv2.warpAffine(frame, rot_mat, (bound_w, bound_h), flags=cv2.INTER_CUBIC)
                y_center_warped = (pts_warped[48]['y'] + pts_warped[55]['y']) / 2
                x_center_warped = (pts_warped[46]['x'] + pts_warped[47]['x']) / 2
                mouth_center_warped = np.array((x_center_warped, y_center_warped))
                nose_center_warped = np.array((pts_warped[34]['x'], pts_warped[34]['y']))
                mn_dist_warped = np.linalg.norm(mouth_center_warped - nose_center_warped)
                lr_dist_warped = (1+CR_RATE)*pts_warped[47]['x'] - (1-CR_RATE)*pts_warped[46]['x']
                if lr_dist_warped > 0.75 * w:
                    lr_dist_warped = (1+CR_SMALL_RATE)*pts_warped[47]['x'] - (1-CR_SMALL_RATE)*pts_warped[46]['x']
                #print ('mouth-nose:', 2*mn_dist)
                #print ('left-right:', (1+CR_RATE)*pts_warped[47]['x'] - (1-CR_RATE)*pts_warped[46]['x'])
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
                cv2.imwrite('%s/%d.jpg' % (save_path_warp, cnt), mouth_warped)    #保存包装修剪后的嘴部图片
                #cv2.imwrite('%s/%s%d.jpg' % (save_path_warp, 'bad/' if bad_flag else '', cnt), mouth_warped)
                '''
                # Draw the cropped region on the warped frame.
                crop = warped.copy()
                for pt in pts_warped:
                    cv2.circle(crop, (int(pt['x']), int(pt['y'])), 4, (0, 255, 0), -1)
                cv2.circle(crop, (int(x_center_warped), int(y_center_warped)), 4, (255, 0, 0), -1)
                cv2.circle(crop, (int(pts_warped[34]['x']), int(pts_warped[34]['y'])), 4, (255, 0, 0), -1)
                cv2.rectangle(crop, (x_st_warped, y_st_warped), (x_ed_warped, y_ed_warped), (0, 0, 255), 3, 8, 0)
                os.makedirs(prev_crop_path, exist_ok = True)
                cv2.imwrite('%s/%d.jpg' % (prev_crop_path, cnt), crop)
                '''
                duration = time.time() - start_t
            else:
                duration = 0.0
            printProgressBar(cnt, seq_l, prefix = '[%d/%d] %s:' % (glob_cnt, total, sub_path), suffix = 'Iter: %.2fs' % duration, length = 50)
        with open(noface_path, "w") as f:
            for idx in range(1, seq_l + 1):
                expected_path = "%s/%d.json" % (save_path_info, idx)
                if not os.path.exists(expected_path):
                    f.write(str(idx) + '\n')
        print ("--> Max weighted angle: %d" % largest_angle)
