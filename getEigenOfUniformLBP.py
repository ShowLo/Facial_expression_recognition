# -*- coding: utf-8 -*-
"""
Created on Sat May 13

@author: 陈佳榕

参考了'huangtao'的Github开源代码 -- https://github.com/michael92ht/LBP
"""

import cv2
import numpy as np
from PIL import Image
from getUniformLBPdict import getUniformDict

#图像的LBP原始特征计算算法：将图像某个位置的像素与周围8个像素逐一比较
#比这个位置灰度值大的点赋值为1，否则为0，然后将这些1,0组成一个二进制序列并返回
def caluteBasicLBP(image, i, j):
    """
    :param image: numpy.ndarray, 图像矩阵
    :param i: int, 行坐标
    :param j: int, 列坐标
    :rtype :numpy.ndarray, 图像的LBP原始特征
    """
    basic_lbp = [];
    if image[i - 1,j - 1] > image[i,j]:
        basic_lbp.append(1);
    else:
        basic_lbp.append(0);
    if image[i - 1,j] > image[i,j]:
        basic_lbp.append(1);
    else:
        basic_lbp.append(0);
    if image[i - 1,j + 1] > image[i,j]:
        basic_lbp.append(1);
    else:
        basic_lbp.append(0);
    if image[i,j - 1] > image[i,j]:
        basic_lbp.append(1);
    else:
        basic_lbp.append(0);
    if image[i,j + 1] > image[i,j]:
        basic_lbp.append(1);
    else:
        basic_lbp.append(0);
    if image[i + 1,j - 1] > image[i,j]:
        basic_lbp.append(1);
    else:
        basic_lbp.append(0);
    if image[i + 1,j] > image[i,j]:
        basic_lbp.append(1);
    else:
        basic_lbp.append(0);
    if image[i + 1,j + 1]>image[i,j]:
        basic_lbp.append(1);
    else:
        basic_lbp.append(0);
    return basic_lbp;

#获取图像的LBP基本模式特征
def basicLBP(image):
    """
    :param image: numpy.ndarray, 图像矩阵
    :rtype :numpy.ndarray, LBP基本模式特征
    """
    basicMatrix = np.zeros(image.shape, np.uint8);
    (height, width) = image.shape;
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            basic_lbp = caluteBasicLBP(image, i, j);    #计算每个像素点的基本LBP特征
            offset = 0;
            sum = 0;
            for item in basic_lbp:
                sum += item << offset;                  #并将这个用二进制序列表示的特征转为对应的十进制数
                offset += 1;
            basicMatrix[i, j] = sum;                    #放入基本模式特征矩阵的相应位置
    return basicMatrix;

#计算某个数的二进制数中1的个数
def count1num(n):
    """
    :param n: int, 某个整数
    :rtype :int, 这个数的二进制数中1的个数
    """
    count = 0;
    while(n):
        n &= n - 1;                                     #与右移移位的自己做与运算会消掉一个1
        count += 1;
    return count;
    
#获取图像的LBP等价模式特征
def uniformLBP(image):
    """
    :param image:np.ndarray, 图像矩阵
    """
    uniformDict = getUniformDict();                     #先获取LBP等价模式的58种特征值对应的字典
    
    uniformLBPmatrix = np.zeros(image.shape, np.uint8);
    basicLBPmatrix = basicLBP(image);                   #先获取图像的基本模式特征
    (height, width) = image.shape;
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            leftShiftNum = basicLBPmatrix[i, j] << 1;   #将数左移
            if leftShiftNum > 255:                      #移出第八位的要移回第一位
                leftShiftNum -= 255;
            #一个数左移之后与自身作异或运算，结果的二进制位上1的个数表明了这个数的二进制数的跳变次数
            jumpNum = count1num(basicLBPmatrix[i, j] ^ leftShiftNum);
            if jumpNum <= 2:                            #小于2的属于基本的58种模式之一
                uniformLBPmatrix[i, j] = uniformDict[basicLBPmatrix[i, j]];
            else:                                       #否则属于第59种模式
                uniformLBPmatrix[i, j] = 58;
    return uniformLBPmatrix;
    

#获得一张灰度图片的LBP等价模式的特征向量
def getEigenOfUniformLBP(imagePath):
    """
    :param imagePath:string, 图片存放位置
    """
    
    image = np.array(Image.open(imagePath));            #读取图片
    uniformLBPmatrix = uniformLBP(image);               #获得其LBP等价模式特征
    #计算LBP等价模式特征的直方图，归一化之后作为其特征向量返回
    hist = cv2.calcHist([uniformLBPmatrix],[0],None,[59],[0,59]);
    cv2.normalize(hist,hist);
    return hist.flatten();
    