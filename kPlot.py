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


def kdataplot(df, code, raw=False, bi=True, xd=True, ktype='D'):
    # 获取笔数据
    df11 = df[df['bipoint'] != 0]
    # 获取线段数据
    df12 = df[df['xdpoint'] != 0]

    if raw == True:
        # 可视化
        # 绘制Raw图
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        if ktype == 'D':
            xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in df.date]
        else:
            xs = [datetime.strptime(d, '%Y-%m-%d %H:%M').date() for d in df.date]
        plt.figure(figsize=(20, 10))
        plt.plot(xs, df['high'])
        plt.plot(xs, df['high'], 'r+')
        plt.title(code)
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记
        # plt.show()
        filename = "C:/PythonProject/KMerger/Data/" + code + "/" + code + "_" + ktype + "_raw.png"
        plt.savefig(filename)
        plt.close()

    if bi == True:
        # 绘制笔图
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        if ktype == 'D':
            xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in df11.date]
        else:
            xs = [datetime.strptime(d, '%Y-%m-%d %H:%M').date() for d in df11.date]
        plt.figure(figsize=(20, 10))
        plt.plot(xs, df11['high'])
        plt.plot(xs, df11['high'], 'r+')
        plt.title(code)
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记
        # plt.show()
        filename = 'C:/PythonProject/KMerger/Data/' + code + "/" + code + "_" + ktype + "_bi.png"
        plt.savefig(filename)
        plt.close()

    if xd == True:
        # 绘制线段图
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        if ktype == 'D':
            xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in df12.date]
        else:
            xs = [datetime.strptime(d, '%Y-%m-%d %H:%M').date() for d in df12.date]
        plt.figure(figsize=(20, 10))
        plt.plot(xs, df12['high'])
        plt.plot(xs, df12['high'], 'r+')
        plt.title(code)
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记
        # plt.show()
        filename = 'C:/PythonProject/KMerger/Data/' + code + "/" + code + "_" + ktype + "_xd.png"
        # plt.show()
        plt.savefig(filename)
        plt.close()


def testkdataplot(df, code):
    # 获取笔数据
    df11 = df[df['bipoint'] != '']
    # 获取线段数据
    df12 = df[df['xdpoint'] != '']

    # 可视化
    # 绘制Raw图
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    xs = [datetime.strptime(d, '%Y-%m-%d %H:%M').date() for d in df.date]
    # plt.figure(figsize=(20,10))
    plt.plot(xs, df['high'])
    plt.plot(xs, df['high'], 'r+')
    plt.title(code)
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.show()
    # filename = "C:/PythonProject/KMerger/Data/" + code + "_D_raw.png"
    # plt.savefig(filename)
    # plt.close()

    # 绘制笔图
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    xs = [datetime.strptime(d, '%Y-%m-%d %H:%M').date() for d in df11.date]
    # plt.figure(figsize=(20,10))
    plt.plot(xs, df11['high'])
    plt.plot(xs, df11['high'], 'r+')
    plt.title(code)
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.show()
    # filename = 'C:/PythonProject/KMerger/Data/' + code + '_D_bi.png'
    # plt.savefig(filename)
    # plt.close()

    # 绘制线段图
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    # plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    # plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())
    # plt.gca().tick_params(pad=10)
    xs = [datetime.strptime(d, '%Y-%m-%d %H:%M').date() for d in df12.date]
    # plt.figure(figsize=(20,10))
    plt.plot(xs, df12['high'])
    plt.plot(xs, df12['high'], 'r+')
    plt.title(code)
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.show()
    # filename = 'C:/PythonProject/KMerger/Data/' + code + '_D_xd.png'
    # plt.savefig(filename)
    # plt.close()


def plotCodeListAndSave(codelist=None, kTypeList=['D', '30'], threadName=None):
    startTime = time.time()
    print("%s plotCodeListAndSave start to Run\n" % threadName)
    # 保存文件的目录
    basepath = 'C:/PythonProject/KMerger/Data/'
    n = 0
    for code in codelist:  # 每个code一个个处理
        for ktype in kTypeList:
            outfilename = basepath + code + '/' + code + '_' + ktype + '_out.csv'
            # print(filename + '\n')
            # try:
            df = pd.read_csv(outfilename, index_col=0)
            df = df.fillna(0)
            maxindex = df.index.max()
            df['bipoint'][maxindex] = 'end'
            df['xdpoint'][maxindex] = 'end'
            # except:
            # 读取文件失败，则直接返回
            # print("fail to read file %s, ktype is %s\n"%(outfilename,ktype))
            # return
            kdataplot(df, code, raw=False, bi=True, xd=True, ktype=ktype)

        n = n + 1
        if n % 5 == 0:
            print("%s finished %d" % (threadName, n))
    print("%s plotCodeListAndSave Finished,cost time is %d" % (threadName, time.time() - startTime))


###多线程处理所有股票的数据将处理后的结果写入到文件中_out
def plotAllData(threadnum=10, kTypeList=['D', '30']):
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
        _thread.start_new_thread(plotCodeListAndSave, (codelist, ['D', '30'], "Thread %s" % i))

