# -*- coding: utf-8 -*-
"""
Created on Sat May 13

@author: 陈佳榕
"""

import numpy as np
import os
from PIL import Image
from skimage.feature import hog

#获得表情特征向量并保存
def saveAndSortEigenHog(filePath = 'cutFace', pathSave = 'sortedEigenHog', trainRate = 0.8):
    """
    :param filePath: string, 图片存放位置，包含8个文件夹
    :param pathSave: string, 存放已整理好的特征向量的文件夹
    :param trainRate: float, 训练率
    """
    
    savePath = os.path.join('.', pathSave);
    if not os.path.exists(savePath):
        os.makedirs(savePath);

    trainNum = testNum = 0;
    eigenVectorTrain = [];
    eigenVectorTest = [];
    labelTrain = [];
    labelTest = [];
    dirs = os.listdir(filePath);                            #列出子目录
    for subDir in dirs:                                     #遍历子目录
        subPath = os.path.join(filePath, subDir);
        files = os.listdir(subPath);                        #列出子目录下的所有文件
        
        fileNum = len(files);
        countNum = int(fileNum * trainRate);
        trainNum = trainNum + countNum;
        testNum = testNum + fileNum - countNum;
        
        count = 0;
        
        for file in files:                                  #获取每个图像的HOG特征向量及标签
            image = np.array(Image.open(os.path.join(subPath, file)));
            hogFeature = hog(image,orientations=9,pixels_per_cell=(20,20),cells_per_block=(2,2),visualise=False,transform_sqrt=True,feature_vector=True);
            if count <= countNum:
                eigenVectorTrain.append(hogFeature);
                labelTrain.append(subDir[0 : 2]);
            else:
                eigenVectorTest.append(hogFeature);
                labelTest.append(subDir[0 : 2]);
            count = count + 1;
    eigenVectorTrain = np.array(eigenVectorTrain);
    eigenVectorTest = np.array(eigenVectorTest);
    labelTrain = np.array(labelTrain);
    labelTest = np.array(labelTest);
    
    if not os.path.exists(savePath):
        os.makedirs(savePath);
    #保存整理好的特征向量与标签
    np.save(os.path.join(savePath,'eigenVectorTrain.npy'), eigenVectorTrain);
    np.save(os.path.join(savePath,'eigenVectorTest.npy'), eigenVectorTest);
    np.save(os.path.join(savePath,'labelTrain.npy'), labelTrain);
    np.save(os.path.join(savePath,'labelTest.npy'), labelTest);