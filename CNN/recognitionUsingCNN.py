# -*- coding: utf-8 -*-
"""
@author: 陈佳榕
"""

import os
import numpy as np
import random
from PIL import Image
import tensorflow as tf

imagePath = '../FaceEmotionRecognition/images'
folders = os.listdir(imagePath)
folderNum = len(folders)
trainImageData = []
testImageData = []
trainLabelData = []
testLabelData = []
labelDict = {}

#训练比例
trainRate = 0.8
#网络输入的图片大小
imageHeight, imageWidth = 48, 48

#先读入图片并进行一些预处理
for i in range(folderNum):
    labelDict[i] = folders[i]
    files = os.listdir(os.path.join(imagePath, folders[i]))
    filesNum = len(files)
    trainNum = int(filesNum * trainRate)
    trainImage = np.zeros((trainNum, imageHeight, imageWidth), dtype = np.float32)
    testImage = np.zeros((filesNum - trainNum, imageHeight, imageWidth), dtype = np.float32)
    trainLabel = np.zeros((trainNum, folderNum), dtype = np.float32)
    testLabel = np.zeros((filesNum - trainNum, folderNum) ,dtype = np.float32)
    random.shuffle(files)
    for j in range(filesNum):
        if j < trainNum:
            #读入图片并resize成网络输入所需大小
            image = np.array((Image.open(os.path.join(imagePath, folders[i], files[j]))).resize((imageHeight, imageWidth)), dtype = np.float32)
            #归一化
            trainImage[j] = image / image.max()
            trainLabel[j, i] = 1
        else:
            image = np.array((Image.open(os.path.join(imagePath, folders[i], files[j]))).resize((imageHeight, imageWidth)), dtype = np.float32)
            testImage[j - trainNum] = image / image.max()
            testLabel[j - trainNum, i] = 1
    trainImageData.append(trainImage)
    testImageData.append(testImage)
    trainLabelData.append(trainLabel)
    testLabelData.append(testLabel)
trainImages = np.vstack(trainImageData)
testImages = np.vstack(testImageData)
trainLabels = np.vstack(trainLabelData)
testLabels = np.vstack(testLabelData)

totalTrainNum = len(trainImages)
totalTestNum = len(testImages)
batch_size = 1024
trainBatchNum = totalTrainNum // batch_size
testBatchNum = totalTestNum // batch_size
train_epoch = 5000
learning_rate = 1e-4

#网络参数
n_classes = folderNum
dropout = 0.6

#tensorflow输入
x = tf.placeholder(tf.float32, shape = [None, imageHeight, imageWidth])
y = tf.placeholder(tf.float32, shape = [None, n_classes])
keep_prob = tf.placeholder(tf.float32)      #dropout

#二维卷积
def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
#二维池化
def max_pool2d(x, height = 3, width = 3, stride = 2):
    return tf.nn.max_pool(x, ksize = [1, height, width, 1], strides = [1, stride, stride, 1], padding = 'SAME')
    
#网络结构
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape = [-1, 48, 48, 1])
    #第一个卷积层
    conv1 = conv2d(x, weights['w1_c'], biases['b1_c'])
    #紧接着的池化层
    pool1 = max_pool2d(conv1)
    
    #第二个卷积层
    conv2 = conv2d(pool1, weights['w2_c'], biases['b2_c'])
    #紧接着的池化层
    pool2 = max_pool2d(conv2)
    
    #第三个卷积层
    conv3 = conv2d(pool2, weights['w3_c'], biases['b3_c'])
    #紧接着的池化层
    pool3 = max_pool2d(conv3)
    
    #全连接层
    fc1 = tf.reshape(pool3, [-1, weights['w1_d'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w1_d']), biases['b1_d'])
    fc1_out = tf.nn.dropout(tf.nn.relu(fc1), dropout)
    
    fc2 = tf.reshape(fc1_out, [-1, weights['w2_d'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['w2_d']), biases['b2_d'])
    fc2_out = tf.nn.dropout(tf.nn.relu(fc2), dropout)
    
    #输出
    out = tf.add(tf.matmul(fc2_out, weights['out']), biases['out'])
    
    return out
    
#网络参数
with tf.name_scope('weights'):
    weights = {
        #几个卷积层的卷积核
        'w1_c' : tf.Variable(tf.random_normal([5, 5, 1, 32])),
        'w2_c' : tf.Variable(tf.random_normal([4, 4, 32, 32])),
        'w3_c' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
        #全连接层
        'w1_d' : tf.Variable(tf.random_normal([6 * 6 * 64, 2048])),
        'w2_d' : tf.Variable(tf.random_normal([2048, 1024])),
        #输出层
        'out' : tf.Variable(tf.random_normal([1024, n_classes]))
        }
with tf.name_scope('biases'):
    biases = {
        'b1_c' : tf.Variable(tf.random_normal([32])),
        'b2_c' : tf.Variable(tf.random_normal([32])),
        'b3_c' : tf.Variable(tf.random_normal([64])),
        'b1_d' : tf.Variable(tf.random_normal([2048])),
        'b2_d' : tf.Variable(tf.random_normal([1024])),
        'out' : tf.Variable(tf.random_normal([n_classes]))
        }
    
#训练模型
pred = conv_net(x, weights, biases, keep_prob)
#定义代价和优化器
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
    tf.summary.scalar('loss', loss)
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

#准确率
with tf.name_scope('correct_pred'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
        
#初始化
init = tf.global_variables_initializer()

trainIndex = np.arange(totalTrainNum)
testIndex = np.arange(totalTestNum)

maxAcc = 0.7

saver = tf.train.Saver()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./summary', sess.graph)
    sess.run(init)
    for epoch in range(0, train_epoch):
        totalTestLoss = 0
        totalTestAcc = 0
        
        #训练
        for i in range(0, trainBatchNum + 1):
            sampleIndex = trainIndex[i * batch_size : min((i + 1) * batch_size, totalTrainNum)]
            batch_x = trainImages[sampleIndex]
            batch_y = trainLabels[sampleIndex]
            sess.run(optimizer, feed_dict = {x : batch_x, y : batch_y, keep_prob : dropout})
            
            cost, acc = sess.run([loss, accuracy], feed_dict = {x : batch_x, y : batch_y, keep_prob : 1.})
            print('Epoch:' + str(epoch + 1) + ', Batch : ' + str(i) + ', Train Loss = ' + '{:.3f}'.format(cost) + ', Train Accuracy = ' + '{:.3f}'.format(acc))
            
        #测试
        '''for i in range(0, testBatchNum + 1):
            sampleIndex = testIndex[i * batch_size : min((i + 1) * batch_size, totalTestNum)]
            batch_x = testImages[sampleIndex]
            batch_y = testLabels[sampleIndex]
            testLoss, testAcc = sess.run([loss, accuracy], feed_dict = {x : batch_x, y : batch_y, keep_prob : 1.})
            totalTestLoss = totalTestLoss + testLoss * len(sampleIndex)
            totalTestAcc = totalTestAcc + testAcc * len(sampleIndex)
        totalTestLoss = totalTestLoss / totalTestNum
        totalTestAcc = totalTestAcc / totalTestNum'''
        result, totalTestLoss, totalTestAcc = sess.run([merged, loss, accuracy], feed_dict = {x : testImages, y : testLabels, keep_prob : 1.})
        writer.add_summary(result, epoch)
        
        print('Epoch:' + str(epoch + 1) + ', Test Loss = ' + '{:.3f}'.format(totalTestLoss) + ', Test Accuracy = ' + '{:.3f}'.format(totalTestAcc))
        
        if totalTestAcc > maxAcc:
            maxAcc = totalTestAcc
            save_path = saver.save(sess, 'myNet/save_net.ckpt')
            print('save to path : ', save_path)