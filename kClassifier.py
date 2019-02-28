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

def kClassifier():
    ###获取股票代码列表
    stock_info = ts.get_stock_basics()
    stock_code_list = stock_info.index.tolist()
    # 保存文件的目录
    basepath = 'C:/PythonProject/KMerger/Data/'

    justBottonList = []
    justTopList = []

    for code in stock_code_list:
        # 先从当前的结果中来读
        try:
            filename = basepath + code + '/' + code + '_' + 'D_out.csv'
            df = pd.read_csv(filename,dtype={'xdobject': object, 'biobject': object, 'xddistance': np.float64})  ###直接读取计算后的结果
            df = df.fillna(0)
            indexlist = df.index.tolist()
            df1 = df[df['bipoint'] != 0]
            df2 = df[df['xdpoint'] != 0]
            p = df['bipoint'][indexlist[len(indexlist) - 2]]
            p1 = df['bipoint'][indexlist[len(indexlist) - 3]]
            p2 = df['bipoint'][indexlist[len(indexlist) - 4]]
            if p == 'Botton' and p1 == 0 and p2 == 0:
                justBottonList.append(code)
                print("find code %s is botton"%code)
            elif p == 'Top' and p1 == 0 and p2 == 0:
                justTopList.append(code)
                print("find code %s is top"%code)
        except:
            pass
        

    print("just botton list is %s, size is %s\n"%(justBottonList,len(justBottonList)))
    print("just top list is %s, size is %s\n"%(justTopList,len(justTopList)))
    

if __name__ == "__main__":
    kClassifier()
