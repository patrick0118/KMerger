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
import threading
import multiprocessing


###start是从哪个index的序号行合并处理的意思
###返回被处理过的df
###start 必须大于2
# K线进行合并
def daykMerger(df, start=0):
    starttime = time.time()
    # K线进行合并参数初始化，这里如果是增量计算，这里的初始合并方向需要重新计算
    indexlist = df.index
    length = len(indexlist)
    if start == 0:
        flag = 'up'
    else:  ##增量的计算的时候需要用到
        indexm = indexlist[start - 1]
        indexm1 = indexlist[start - 2]
        if df['high'][indexm] >= df['high'][indexm1] and df['low'][indexm] >= df['low'][indexm1]:
            flag = 'up'
        if df['high'][indexm] <= df['high'][indexm1] and df['low'][indexm] <= df['low'][indexm1]:
            flag = 'down'
    # print("start flog is %s"%flag)
    # 初始化对应参数
    m = start
    while m < len(indexlist) - 2:  ####TBD ,需要检查这个-2是否OK？
        indexlist = df.index
        indexm = indexlist[m]
        indexm1 = indexlist[m + 1]
        df11 = df.loc[[indexm, indexm1], :]
        # print("indexm is %s, indexm1 = %s, df11 is %s" % (indexm, indexm1, df11))
        # if these two are contain relation
        if (df['high'][indexm] >= df['high'][indexm1] and df['low'][indexm] <= df['low'][indexm1]) or (
                df['high'][indexm] <= df['high'][indexm1] and df['low'][indexm] >= df['low'][indexm1]):
            # print('contain at %d, %d, %d, %d %s'% (m,m+1,indexm,indexm1,flag))
            if flag == 'up':
                df.loc[indexm, 'high'] = df11['high'].max()
                df.loc[indexm, 'low'] = df11['low'].max()
            if flag == 'down':
                df.loc[indexm, 'high'] = df11['high'].min()
                df.loc[indexm, 'low'] = df11['low'].min()
            df.loc[indexm, 'volume'] = df11['volume'].sum()  ##将合并后的量改成最后总量
            df.loc[indexm, 'date'] = df.loc[indexm1, 'date']  ##将合并后的日期改成最后一个日期
            df.drop(indexm1, inplace=True)
            # print("drop %s" % indexm1)
        else:
            if df['high'][indexm] >= df['high'][indexm1] and df['low'][indexm] >= df['low'][indexm1]:
                flag = 'down'
            if df['high'][indexm] <= df['high'][indexm1] and df['low'][indexm] <= df['low'][indexm1]:
                flag = 'up'
            m = m + 1

    ##提取笔 point
    df['bipoint'] = ''
    df['xdpoint'] = ''
    length = len(df.index)
    for n in range(1, len(df.index) - 1):
        indexlist = df.index
        df11 = df.loc[indexlist[n - 1]:indexlist[n + 1], :]
        # if n point is the top point
        if df11['high'][indexlist[n]] == df11['high'].max() and df11['low'][indexlist[n]] == df11['low'].max():
            df.loc[indexlist[n], 'bipoint'] = 'Top'
        # if n point is the botton point
        if df11['high'][indexlist[n]] == df11['high'].min() and df11['low'][indexlist[n]] == df11['low'].min():
            df.loc[indexlist[n], 'bipoint'] = 'Botton'

        # 如果是第二个不是空,需要将第一个点设置为一个临时的顶分型的点
        if n == 1 and df['bipoint'][indexlist[1]] == '':
            if df['bipoint'][indexlist[n]] == 'Top':
                df.loc[indexlist[0], 'bipoint'] = 'Botton'
            else:
                df.loc[indexlist[0], 'bipoint'] = 'Top'

    # 如果倒数第二天不是顶分型或者底分型，则需要将最后一个设置为临时的分型，已配对
    # if df['bipoint'][indexlist[length-2]] == '':
    #    if (flag == 'up'):
    #        df.loc[df.index[length-1],'bipoint'] = 'Botton'
    #    else:
    #        df.loc[df.index[length-1],'bipoint'] = 'Top'
    # print("totalsize is %s, n is %d" %(len(df.index),n))
    # print(df)

    # 提取线段 point
    m = 1
    while m <= len(df.index) - 1:
        indexlist = df.index
        df11 = df.loc[indexlist[m - 1]:indexlist[m], :]
        if df11['bipoint'][indexlist[m - 1]] != '' and df11['bipoint'][indexlist[m]] != '':
            df.loc[indexlist[m - 1], 'xdpoint'] = ''
            df.loc[indexlist[m], 'xdpoint'] = ''
            m = m + 2
        else:
            df.loc[indexlist[m - 1], 'xdpoint'] = df.loc[indexlist[m - 1], 'bipoint']
            df.loc[indexlist[m], 'xdpoint'] = df.loc[indexlist[m], 'bipoint']
            m = m + 1

    # 提取每个Bi的距离
    df['bidistance'] = ''  # 增加新的一列
    df12 = df[df['bipoint'] != '']  # 过滤出来所有线段列不为空的
    indexlist = df12.index
    for n in range(1, len(indexlist)):
        dishigh = round((df12['high'][indexlist[n]] - df12['high'][indexlist[n - 1]]) / df12['high'][indexlist[n - 1]],
                        2)
        dislow = round((df12['low'][indexlist[n]] - df12['low'][indexlist[n - 1]]) / df12['low'][indexlist[n - 1]], 2)
        disave = round((dishigh + dislow) / 2, 2)
        df.loc[indexlist[n], 'bidistance'] = disave

    # 提取每个线段之间的距离
    df['xddistance'] = ''  # 增加新的一列
    df12 = df[df['xdpoint'] != '']  # 过滤出来所有线段列不为空的
    indexlist = df12.index
    for n in range(1, len(indexlist)):
        dishigh = round((df12['high'][indexlist[n]] - df12['high'][indexlist[n - 1]]) / df12['high'][indexlist[n - 1]],2)
        dislow = round((df12['low'][indexlist[n]] - df12['low'][indexlist[n - 1]]) / df12['low'][indexlist[n - 1]], 2)
        disave = round((dishigh + dislow) / 2, 2)
        df.loc[indexlist[n], 'xddistance'] = disave
    spendtime = time.time() - starttime
    #print("process one k spend time is %s"%spendtime)
    return df


###处理一个codelist的数据并写入到out结果中
def processCodeListAndSave(codelist=None, kTypeList=['D', '30'], threadName=None):
    startTime = time.time()
    print("%s processCodeList and Save start to Run\n" % threadName)
    # 保存文件的目录
    basepath = 'C:/PythonProject/KMerger/Data/'
    n = 0
    for code in codelist:  # 每个code一个个处理
        for ktype in kTypeList:
            outfilename = basepath + code + '/' + code + '_' + ktype + '_out.csv'
            filename = basepath + code + '/' + code + '_' + ktype + '.csv'
            # print(filename + '\n')
            try:
                df = pd.read_csv(filename, index_col=0)
                df = df[df['date'] > '2017-04-01']
            except:
                # 读取文件失败，则直接返回
                print("fail to read file %s, ktype is %s\n" % (filename, ktype))
                return
            df = daykMerger(df)
            df.to_csv(outfilename, mode='w', float_format='%.3f', index_label='Index')

        n = n + 1
        if n % 5 == 0:
            print("%s finished %d" % (threadName, n))
    print("%s processCodeList and Save start Finished,cost time is %d" % (threadName, time.time() - startTime))


###多线程处理所有股票的数据将处理后的结果写入到文件中_out
def processAllData(threadnum=10, kTypeList=['D', '30']):
    ###获取股票代码列表
    stock_info = ts.get_stock_basics()
    code_list = stock_info.index.tolist()
    # 创建多线程处理
    step = 3500 / threadnum
    for i in range(threadnum):
        start = int(i * step)
        end = int((i + 1) * step)
        # 将每个线程处理的code的范围确定下来
        if i == threadnum - 1:
            codelist = code_list[start:]
        else:
            codelist = code_list[start:end]
        #_thread.start_new_thread(processCodeListAndSave, (codelist, ['D', '30'], "Thread %s" % i))
        #my_thread = threading.Thread(target=processCodeListAndSave, args=(codelist, ['D', '30'], "Thread %s" % i))
        #my_thread.start()
        multiprocessing.Process(target = processCodeListAndSave, args = (codelist, ['D', '30'], "Thread %s" % i)).start()
        time.sleep(1)

    ###处理一个codelist的数据并写入到out结果中
def incProcessCodeListAndSave(codelist=None, kTypeList=['D', '30'], threadName=None):
    startTime = time.time()
    print("%s incProcessCodeListAndSave start to Run\n" % threadName)
    # 保存文件的目录
    basepath = 'C:/PythonProject/KMerger/Data/'
    n = 0
    for code in codelist:  # 每个code一个个处理
        for ktype in kTypeList:
            outfilename = basepath + code + '/' + code + '_' + ktype + '_out.csv'
            filename = basepath + code + '/' + code + '_' + ktype + '.csv'
            # print(filename + '\n')
            try:
                df = pd.read_csv(filename, index_col=0)
            except:
                # 读取文件失败，则直接返回
                print("fail to read file %s, ktype is %s\n" % (filename, ktype))
                return

            # 读取原来已经存在的out的文件，将已经处理的不再处理
            dfout = pd.read_csv(outfilename, index_col=0)
            maxdate = dfout['date'].max()  ##取已经处理过的数据中的最近的一个时间
            print("maxdate is %s" % maxdate)
            # 将读取的原始数据df进行过滤，将已经处理过的数据过滤掉 TBD
            df = df[df['date'] > maxdate]
            # print(df)
            m = len(dfout.index.tolist())
            # 将dfout和df这两个数据进行合并  TBD
            dfout = dfout.append(df)
            print(dfout)

            # 调用daykMerger方法，并且指定开始计算的位置进行开始计算  怎么指定对应的点TBD
            dfout = daykMerger(dfout, m - 1)
            dfout.to_csv(outfilename, mode='w', float_format='%.3f', index_label='Index')

        n = n + 1
        if n % 10 == 0:
            print("%s finished %d" % (threadName, n))
    print("%s processCodeList and Save start Finished,cost time is %d" % (threadName, time.time() - startTime))


###多线程处理所有股票的数据将处理后的结果写入到文件中_out
def incProcessAllData(threadnum=10, kTypeList=['D', '30']):
    ###获取股票代码列表
    stock_info = ts.get_stock_basics()
    code_list = stock_info.index.tolist()
    # 创建多线程处理
    step = 3500 / threadnum
    for i in range(threadnum):
        start = int(i * step)
        end = int((i + 1) * step)
        # 将每个线程处理的code的范围确定下来
        if i == threadnum - 1:
            codelist = code_list[start:]
        else:
            codelist = code_list[start:end]
        _thread.start_new_thread(incProcessCodeListAndSave, (codelist, ['D', '30'], "Thread %s" % i))
        time.sleep(1)

    # processCodeListAndSave(codelist=['002811'],kTypeList=['30'])

if __name__ == "__main__":
    print("start to process All Data")
    processAllData()
    print("finish to process All Data")
