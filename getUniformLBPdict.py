# -*- coding: utf-8 -*-
"""
Created on Sat May 13

@author: 陈佳榕

参考了'huangtao'的Github开源代码 -- https://github.com/michael92ht/LBP
"""

#求一个8位二进制数旋转一周得到的8个值
def get8valueOfRotate(num, values):
    """
    :param num: int, 8位二进制数
    :param values: list, 由旋转得到的各种排列的8位二进制数
    """
    for i in range(0,8):
        offset = 0;
        value = 0;
        for j in range(i, 8):           #以i为开始旋转的位置
            value += num[j] << offset;
            offset += 1;
        for k in range(0, i):
            value += num[k] << offset;  #接着旋转
            offset += 1;
        values.append(value);

#求等价模式的58种特征值对应的字典
def getUniformDict():
    """
    :rtype :dict, 等价模式的58种特征值为键，种类序号为值的字典
    """
    values = [];
    for i in range(1, 8):               #模拟连续出现1~7个1
        num = [0] * 8;
        j = 0;
        while j < i:
            num[j] = 1;
            j += 1;
        get8valueOfRotate(num,values);  #然后求其旋转一周得到的8个值
    values.sort();
    dict = {0:0,255:57};                #0和257对应没有跳变的情况
    type = 1;
    for v in values:                    #构造所需的字典
        dict[v] = type;                 #键为最多有两次跳变的二进制序列对应的值，值为对应的种类号
        type += 1;
    return dict;