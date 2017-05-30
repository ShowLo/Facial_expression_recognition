# -*- coding: utf-8 -*-
"""
Created on Sat May 13

@author: 陈佳榕
"""

import numpy as np
import os
from PIL import Image
from getEigenOfUniformLBP import getEigenOfUniformLBP

#获得各个分块的LBP等价模式的经过归一化的统计直方图并组合成表情特征向量
def getBlockEigen(image, widthBlock = 10, heightBlock = 10):
    """
    :param image: numpy.ndarray, 图像矩阵
    :param widthBlock: int, 宽度方向上分块数
    :param heightBlock: int, 高度方向上分块数
    :rtype: numpy.ndarray, 由各个分块的特征向量组成的整个特征向量
    """
    aEigenLen = 59;
    eigenLength = aEigenLen * widthBlock * heightBlock;     #由各个分块的特征向量组成的整个特征向量的长度
    (width, height) = image.shape;
    widthBlockLen = width // widthBlock;                    #宽度方向上每个分块的大小
    heightBlockLen = height // heightBlock;                 #高度方向上每个分块的大小
    eigenVector = np.empty([eigenLength], np.float32);
    for h in range(heightBlock):                            #依次获得各个分块的特征向量并组合起来
        for w in range(widthBlock):
            index = (h * widthBlock + w)*aEigenLen;
            eigenVector[index : index + aEigenLen] = getEigenOfUniformLBP(image[h*heightBlockLen : (h+1)*heightBlockLen, w*widthBlockLen : (w+1)*widthBlockLen]);
    return eigenVector;

#获得表情特征向量并保存
def saveEigen(filePath = 'cutFace', pathSave = 'eigenVector', trainRate = 0.8, widthBlock = 10, heightBlock = 10):
    """
    :param filePath: string, 图片存放位置，包含8个文件夹
    :param pathSave: string, 存放特征向量的文件夹
    :param trainRate: float, 训练率
    :param widthBlock: int, 宽度方向上分块数
    :param heightBlock: int, 高度方向上分块数
    """
    aEigenLen = 59;
    
    savePath = os.path.join('.', pathSave);
    if not os.path.exists(savePath):
        os.makedirs(savePath);

    trainNum = testNum = 0;
    dirs = os.listdir(filePath);                            #列出子目录
    for subDir in dirs:                                     #遍历子目录
        subPath = os.path.join(filePath, subDir);
        files = os.listdir(subPath);                        #列出子目录下的所有文件
        
        fileNum = len(files);
        trainNum = trainNum + int(fileNum * trainRate);
        testNum = testNum + fileNum - int(fileNum * trainRate);
        
        eigenVector = np.empty([fileNum, aEigenLen * widthBlock * heightBlock], np.float32);
        index = 0;
        for file in files:                                  #获取每个图像的LBP特征向量
            image = np.array(Image.open(os.path.join(subPath, file)));
            eigenVector[index] = getBlockEigen(image);
            index = index + 1;
        fileNameTrain = os.path.join(savePath, subDir + '_train.npy');
        fileNameTest = os.path.join(savePath, subDir + '_test.npy');
        #分别保存用于训练和测试的特征向量
        np.save(fileNameTrain, eigenVector[0 : int(fileNum * trainRate)]);
        np.save(fileNameTest, eigenVector[int(fileNum * trainRate) : fileNum]);
    np.save('trainAndTestNum.npy',np.array([trainNum, testNum]));