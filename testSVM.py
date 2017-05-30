# -*- coding: utf-8 -*-
"""
Created on Sat May 13

@author: 陈佳榕
"""

import os
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

def testSVM(eigenPath = 'sortedEigen'):
    """
    :param eigenPath: String, 存放整理好的训练用特征向量的文件夹
    """
    eigen = np.load(os.path.join(eigenPath, 'eigenVectorTest.npy'));
    label = np.load(os.path.join(eigenPath, 'labelTest.npy'));
    clf = joblib.load('train_model.m');
    testResult = clf.predict(eigen);
    return sum(label == testResult)*1./len(label);