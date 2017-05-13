# -*- coding: utf-8 -*-
"""
Created on Wed May 10

@author: 陈佳榕

参考了官方文档 -- http://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html
"""
import cv2
import os
import numpy as np

def cutHumanFace(filePath):
    """
    :rtype: list, 包含人脸区域的四个边角点坐标信息
    """
    
    #print(filePath)
    
    image = cv2.imread(filePath);
    #如果是彩色图的话先转为灰度图，以使其可以作为harr分类器的输入
    if image.ndim == 3:
        grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
    else:
        grayImage = image;
    
    #利用opencv默认提供的训练好的分类器数据产生一个harr分类器
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

    #多尺度检测
    faces = cascade.detectMultiScale(grayImage, 1.3, 5);
    #获取人脸区域
    if len(faces):
        (x, y, w, h) = faces[0];
        #识别到了，返回人脸灰度图
        return grayImage[y:y+h, x:x+w];
    else:
        #没识别到
        return np.array([]);