import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import time
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import _thread
import sklearn.metrics as mc

def lgdataprepare(df=None, fdim=5):
    # 提取出来对应的数据
    df = df.fillna(0)
    # print(df)
    df11 = df[df['xddistance'] != 0]
    list = np.array(df11['xddistance']).tolist()
    # print(list)
    m = fdim
    resarray = []
    while m < len(list) - 2:
        if list[m] < 0:
            resarray.append(list[m - fdim + 1:m + 2])
        m = m + 1
    # print(len(resarray))
    return resarray


###进行LR训练
def lr(trainndarray=None, testndarray=None, num=300, fdim=3, increaseRate=0.20):
    lrmodel = LogisticRegression(class_weight={0:1,1:1})
    X = trainndarray[:, :fdim]
    y = trainndarray[:, fdim:]
    y = np.ravel(y)  # 将y转换成一维的vector
    for n in range(len(y)):
        if y[n] > increaseRate:
            y[n] = 1
        else:
            y[n] = 0

    # 准备测试集合
    X_test = testndarray[:, :fdim]
    y_test = testndarray[:, fdim:]
    for n in range(len(y_test)):
        if y_test[n] > increaseRate:
            y_test[n] = 1
        else:
            y_test[n] = 0

    # print(X)
    # print(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
    # print("LR training started")
    lrmodel.fit(X, y)
    for i in range(0, X.ndim):
        Xi = np.reshape(X[i], (1, -1))
        result = lrmodel.predict(Xi)
        if result == [1]:
            print("Predict is 1, actual y is %s" % y[i])
    score = lrmodel.score(X_test,y_test)
    print(" LR predit accuracy is %s when num is %s /n" % (score,num))

    ###Sklearn Metrics
    y_proba = lrmodel.predict_proba(X_test)[:,1]
    print(y_proba)
    y_predict = lrmodel.predict(X_test)
    print(y_predict)
    print(y_test)
    p,r,t = p_r_curve = mc.precision_recall_curve(y_test,y_proba)
    plt.plot(r,p)
    plt.show()

    fpr,tpr,t = mc.roc_curve(y_test,y_predict)
    plt.plot(fpr,tpr)
    plt.show()

    
    
    return lrmodel


def trainmodel(num=300, fdim=3):
    ###获取股票代码列表
    stock_info = ts.get_stock_basics()
    stock_code_list = stock_info.index.tolist()
    # 保存文件的目录
    basepath = 'C:/PythonProject/KMerger/Data/'
    # 对对应的code进行处理
    stock_code_list = stock_code_list[:num]  ###提取其中的num股票数量的数据作为样本

    # 准备训练集合，提取某一段时间内的数据
    lrlist = []
    for code in stock_code_list:
        # 先从当前的结果中来读
        #try:
        filename = basepath + code + '/' + code + '_' + 'D_out.csv'
        df = pd.read_csv(filename,dtype={'xdobject': object, 'biobject': object, 'xddistance': np.float64})  ###直接读取计算后的结果
        df['date'] = pd.to_datetime(df['date'])
        #print(df)
        df = df[df['date'] < '2017-06-30']
        lrlist.extend(lgdataprepare(df, fdim))  ###将获取的用户的数据拼接成一个list
        #except:
        #   # 如果发生了错误，则从原始数据中读取并进行处理
        #    print("%s code does not exist" % code)
        #    # filename = basepath + code + '/' + code + '_' + 'D.csv'
        #    # df = pd.read_csv(filename)
        #    # df = daykMerger(df)
        #    # pass
        # 准备训练集合，提取某一段时间内的数据

    lrtestlist = []
    stock_code_list = stock_code_list[:num]
    for code in stock_code_list:
        # 先从当前的结果中来读
        #try:
        filename = basepath + code + '/' + code + '_' + 'D_out.csv'
        df = pd.read_csv(filename,dtype={'xdobject': object, 'biobject': object, 'xddistance': np.float64})  ###直接读取计算后的结果
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] > '2017-07-01']
        lrtestlist.extend(lgdataprepare(df, fdim))  ###将获取的用户的数据拼接成一个list
        #except:
        #    # 如果发生了错误，则从原始数据中读取并进行处理
        #    print("%s code does not exist" % code)
        #   # filename = basepath + code + '/' + code + '_' + 'D.csv'
        #    # df = pd.read_csv(filename)
        #    # df = daykMerger(df)
        #    # pass

    lrdata = np.array(lrlist)
    lrtestdata = np.array(lrtestlist)
    # 进行LR模型训练
    lrmodel = lr(lrdata, lrtestdata, num, fdim)

    return lrmodel


def iatestmodel(lrmodel, fdim=10):
    # 保存文件的目录
    basepath = 'C:/PythonProject/KMerger/Data/'
    ##使用模型为一个code的数据进行predict
    inputStr = input("please input the code to verfify:")
    while (inputStr != 'z'):
        code = inputStr
        print("start to predict code is %s" % code)
        filename = basepath + code + '/' + code + '_' + 'D.csv'
        df = pd.read_csv(filename)
        df = daykMerger(df)
        df11 = df[df['xddistance'] != '']
        list = np.array(df11['xddistance']).tolist()
        m = fdim - 1
        while m < len(list) - 1:
            print("start to process m is %s" % m)
            if list[m] < 0:
                X = list[m - fdim + 1:m + 1]
                X = np.reshape(X, (1, -1))  ###验证的时候，X必须是二维的数组，所以需要重新将list转变成二位数组
                y = lrmodel.predict(X)
                print(X)
                print("predict y is %s, actual y is %s" % (y, list[m + 1]))
            m = m + 1

        inputStr = input("please input the code to verfify:")


if __name__ == "__main__":
    numlist = [1000]
    dim = [5]
    lrmodel = LogisticRegression()
    for num in numlist:
        print("Num is %s" %num)
        for fdim in dim:
            lrmodel = trainmodel(num,fdim)

# iatestmodel(lrmodel,fdim=5)
