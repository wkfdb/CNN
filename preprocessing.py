# encoding=utf-8
"""
Preprocess the dataset
"""
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from PIL import Image,ImageOps
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

IMAGE_SIZE=48
IMAGE_CHANNELS=3
TRAIN_PATH="/home/administrator/PengXiao/plant/dataset/train"
OUTPUT_NODE=12
#mask掩膜，用于遮蔽部分图像，对于图像预处理，可以提取特征图像
def create_mask_for_plant(image):
    #将RGB颜色空间转换为hsv颜色空间，image_hsv已经是一张新图像了
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    #图像值中低于lower_hsv和高于upper_hsv的图像值变为0，lower_hsv--upper_hsv之间的值变为255
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    #一个椭圆形的结构元素
    #openvc中morphologyEx（）函数是一种形态学变化函数。数学形态学可以理解为一种滤波行为，
    #因此也称为形态学滤波。滤波中用到的滤波器(kernal)，在形态学中称为结构元素。
    #结构元素往往是由一个特殊的形状构成,如线条、矩形、圆等。
    #闭运算(close)：先膨胀后腐蚀的过程。闭运算可以用来排除小黑洞。

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
    #最终提取了椭圆特征？？


def segment_plant(image):
    #预处理
    mask = create_mask_for_plant(image)
    #白色区域保留，黑色区域删除
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def sharpen_image(image):
    #高斯模糊
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    #图像锐化
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def img_segment(image_path):
    #rgb格式打开图像
    image_rgb = Image.open(image_path)
    #np.array()转化为矩阵
    image_segmented = segment_plant(np.array(image_rgb))
    image_sharpen = sharpen_image(image_segmented)
    #预处理加锐化，输出为48*48的图像
    image_sharpen_reshaped = ImageOps.fit(Image.fromarray(image_sharpen), (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS).convert('RGB')
    return image_sharpen_reshaped


def create_dataset(test_percentage, labelBinarizer):
    print("Start to process dataset")
    ori_imgs = []
    ori_label = []
    sub_dirs = [x[0] for x in os.walk(TRAIN_PATH)]
    is_root = True
    for sub_dir in sub_dirs:
        if is_root:
            is_root = False
            continue
        file_list = []
        dir_name = os.path.basename(sub_dir)

        glob_path = os.path.join(sub_dir, '*')
        file_list.extend(glob.glob(glob_path))

        if len(file_list) == 0:
            continue

        for file_name in file_list:
            # new_img = Image.open(file_name)
            # new_img = ImageOps.fit(new_img, (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS).convert('RGB')
            new_img = img_segment(file_name)
            ori_imgs.append(new_img)
            ori_label.append(dir_name)

    imgs = np.array([np.array(im) for im in ori_imgs])
    imgs = imgs.reshape(imgs.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3)
    lb = labelBinarizer.fit(ori_label)
    label = lb.transform(ori_label)

    trainX, testX, trainY, testY = train_test_split(imgs, label, test_size=test_percentage, random_state=42, shuffle=True)

    return trainX, testX, trainY, testY
