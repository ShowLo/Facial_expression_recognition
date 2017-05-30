# -*- coding: utf-8 -*-
"""
Created on Sat May 13

@author: 陈佳榕
"""

import os
import numpy as np

#整理特征向量
def sortEigen(savePath = 'sortedEigen', eigenVectorPath = 'eigenVector', trainRate = 0.8, widthBlock = 10, heightBlock = 10):
    """
    :param savePath: String, 保存整理好的特征向量与标签
    :param eigenVectorPath: String, 保存特征向量的文件夹
    :trainRate: float, 训练比例
    :param widthBlock: int, 宽度方向上分块数
    :param heightBlock: int, 高度方向上分块数
    """
    trainAndTestNum = np.load('trainAndTestNum.npy');
    trainNum = trainAndTestNum[0];                      #用于训练的特征向量数量
    testNum = trainAndTestNum[1];                       #用于测试的特征向量数量
    
    aEigenLen = 59;
    
    eigenVectorTrain = np.empty([trainNum, aEigenLen*widthBlock*heightBlock], np.float32);
    eigenVectorTest = np.empty([testNum, aEigenLen*widthBlock*heightBlock], np.float32);
    labelTrain = np.empty([trainNum], dtype = 'S2');
    labelTest = np.empty([testNum], dtype = 'S2');
    
    trainIndex = testIndex = 0;
    files = os.listdir(eigenVectorPath);
    for file in files:                                  #遍历加载各个特征向量
        fileName = os.path.join(eigenVectorPath, file);
        eigenVector = np.load(fileName);                #加载特征向量
        evLen = len(eigenVector);
        if file[-5] == 'n':                             #用于训练
            eigenVectorTrain[trainIndex : trainIndex + evLen] = eigenVector;
            labelTrain[trainIndex : trainIndex + evLen] = file[0 : 2];
            trainIndex = trainIndex + evLen;
        else:                                           #用于测试
            eigenVectorTest[testIndex : testIndex + evLen] = eigenVector;
            labelTest[testIndex : testIndex + evLen] = file[0 : 2];
            testIndex = testIndex + evLen;
    
    if not os.path.exists(savePath):
        os.makedirs(savePath);
    #保存整理好的特征向量与标签
    np.save(os.path.join(savePath,'eigenVectorTrain.npy'), eigenVectorTrain);
    np.save(os.path.join(savePath,'eigenVectorTest.npy'), eigenVectorTest);
    np.save(os.path.join(savePath,'labelTrain.npy'), labelTrain);
    np.save(os.path.join(savePath,'labelTest.npy'), labelTest);