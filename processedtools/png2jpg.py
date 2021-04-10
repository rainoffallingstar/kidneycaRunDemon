# =========================================================================
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
#import copy


#定义源文件目录，请按目录数启用处理模块
file_dir1 = r'/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/notca/'
file_dir2 = r'/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/notca/'
file_dir3 = r'/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/notca/'
file_dir4 = r'/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/notca/'
file_dir5 = r'/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/notca/'

# 图片文件夹路径1内操作

for img_name in os.listdir(file_dir1):
    img_path = file_dir1 + img_name
    img = cv2.imread(img_path)
    cv2.imwrite(file_dir1 + img_name[0:-4] + '.jpg')
    
print("transfer done in file_dir1")

# 图片文件夹路径2内操作

for img_name in os.listdir(file_dir2):
    img_path = file_dir2 + img_name
    img = cv2.imread(img_path)
    cv2.imwrite(file_dir2 + img_name[0:-4] + '.jpg')
    
print("transfer done in file_dir2")

# 图片文件夹路径3内操作

for img_name in os.listdir(file_dir3):
    img_path = file_dir3 + img_name
    img = cv2.imread(img_path)
    cv2.imwrite(file_dir3 + img_name[0:-4] + '.jpg')
    
print("transfer done in file_dir3")

# 图片文件夹路径4内操作

for img_name in os.listdir(file_dir4):
    img_path = file_dir4 + img_name
    img = cv2.imread(img_path)
    cv2.imwrite(file_dir4 + img_name[0:-4] + '.jpg')
    
print("transfer done in file_dir4")

# 图片文件夹路径5内操作

for img_name in os.listdir(file_dir5):
    img_path = file_dir5 + img_name
    img = cv2.imread(img_path)
    cv2.imwrite(file_dir5 + img_name[0:-4] + '.jpg')
    
print("transfer done in file_dir5")

# 图片文件夹路径1内操作

for img_name in os.listdir(file_dir5):
    img_path = file_dir5 + img_name
    img = cv2.imread(img_path)
    cv2.imwrite(file_dir5 + img_name[0:-4] + '.jpg')
    
print("transfer done in file_dir5")
print("all done,exit")
