# -*- coding: utf-8 -*-
"""
Created on Thurs May 11

@author: 陈佳榕
"""
import cv2
import os
import numpy
import cutHumanFace

def saveCutFace(filePath):
    """
    :param filePath: string, 文件夹路径，包含了八个文件夹，每个文件夹里面都是相应的表情图片
    """
    
    normalizeWidth = normalizeHeight = 300;             #设置归一化的宽度和高度
    
    dirs = os.listdir(filePath);                        #列出子目录
    savePath = os.path.join('.', 'cutFace');
    for subDir in dirs:                                 #遍历子目录
        saveSubPath = os.path.join(savePath, subDir);
        if not os.path.exists(saveSubPath):
            os.makedirs(saveSubPath);                   #保存人脸区域的文件夹
            
        subPath = os.path.join(filePath, subDir);
        files = os.listdir(subPath);                    #列出子目录下的所有文件
        
        for file in files:
            #因为用到的数据中有一些已经是截好了人脸的，需要特殊处理一下，这里并不具有一般性
            #对于这些图片直接保存其灰度图即可
            if file[0] == 'f' or file[0] == 'm':
                image = cv2.imread(os.path.join(subPath, file));
                if image.ndim == 3:
                    cutFace = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
                else:
                    cutFace = image;
                #进行尺度归一化操作并保存灰度图片
                normalizeFace = cv2.resize(cutFace, (normalizeWidth,normalizeHeight), interpolation=cv2.INTER_AREA);
                cv2.imwrite(os.path.join(saveSubPath, file), normalizeFace);
                continue;
            
            #下面的具有一般性，先截出人脸灰度图，进行尺度归一化再保存
            cutFace = cutHumanFace.cutHumanFace(os.path.join(subPath, file));
            if cutFace.size:
                normalizeFace = cv2.resize(cutFace, (normalizeWidth,normalizeHeight), interpolation=cv2.INTER_AREA);
                cv2.imwrite(os.path.join(saveSubPath, file), normalizeFace);