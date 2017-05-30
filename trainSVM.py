# -*- coding: utf-8 -*-
"""
Created on Sat May 13

@author: 陈佳榕
"""

import os
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
    
def trainSVM(eigenPath = 'sortedEigen'):
    """
    :param eigenPath: String, 存放整理好的训练用特征向量的文件夹
    """
    eigen = np.load(os.path.join(eigenPath, 'eigenVectorTrain.npy'));
    label = np.load(os.path.join(eigenPath, 'labelTrain.npy'));
    #用LBP特征进行训练
    #clf = svm.SVC(kernel = 'rbf', C = 3, gamma = 0.06);    #选用高斯核函数，准确率为80.58%
    #clf = svm.LinearSVC(C = 0.27);                         #准确率为80.27%
    
    #用HOG特征进行训练
    #clf = svm.LinearSVC(C = 0.159);                        #准确率为86.14%
    #clf = svm.SVC(kernel = 'rbf', C = 12.1, gamma = 0.06);  #准确率为84.38%
    clf = svm.SVC(kernel = 'linear', C = 0.159);
    clf.fit(eigen, label);                                  #训练SVM
    joblib.dump(clf, 'train_model.m');                      #保存SVM模型