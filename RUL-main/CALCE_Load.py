

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pdb
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import torch
import random
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
Battery_List = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
dir = './dataset'
Rated_Capacity = 1.1
# Battery = {}
# print(load.drop_outliers(np.arange(10),10,2))
# Battery=load.load_data(Battery_List,dir)
# print(Battery)
# pdb.set_trace()
### 这部分函数用于去除掉异常点
def drop_outliers(array, count, bins):
    index = []
    range_arr = np.arange(1, count, bins)
    # [1, 1+bins ,1+2*bins, ...., 1+n*bins...]
    for i in range_arr[:]:
        array_tmp = array[i:i+bins]
        # array_lim = array[range_arr[i]:range_arr[i+1]] 跟上面一个意思
        sigma = np.std(array_tmp)
        mean = np.mean(array_tmp)
        conf_max, conf_min = mean+sigma*2, mean - sigma*2  # 95%confidance
        idx = np.where((array_tmp > conf_min) & (array_tmp < conf_max))[0]
        # np.where(condition)只给condition  返回condition.nanzero()
        # 就是返回满足True的索引
        idx = idx + i  # where取得每个bins的复合条件的值，idx记录array中所有复合条件的值
        index.extend(list(idx))
        # 这里的index_t出来是 list 要用np.array变成数组的形式
    return np.array(index)
        # 变成 array_like的形式


### 读取信息
# Battery_List=['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
# dir = './dataset'
# dir='./batt/CALCE/dataset/'
# dir = '.../batt/CALCE/CALCE/dataset'
# dir = 'E:/DL/1workspace/batt/CALCE/CALCE/dataset'
Battery = {}
def load_data(Battery_List, dir):
# 分别读取4个电池
    for batt in Battery_List:
        print('Load '+batt+'..')
        paths = glob.glob(os.path.abspath(os.path.join(dir, batt, '*.xlsx')))
        # print(paths[0])
        # pdb.set_trace()
        times=[]
        # 这一步取得所有文件下的第一个时间点
        for path in paths:
            df = pd.read_excel(path, sheet_name=1)  #第二个表格
            # print('Load ' + str(path) + ' ...')
            times.append(df['Date_Time'][0])
        # 按时间将文件顺序进行排序
        idx = np.argsort(times)
        paths_sorted = np.array(paths)[idx]
        # 声明各个变量
        # 循环时间 放电容量 内阻 电压 电流
        # 恒流充电时间CCCT 恒压充电时间CVCT
        count = 0  # 用于记录循环次数
        dc_capacities=[]  # 每个循环的放电量
        internal_resistance=[]  # 每个循环的平均内阻
        state_of_health=[]  # 定义为3.8-3.4V的放电量
        CCCT=[]  #恒流充电时间
        CVCT=[]  #恒流放电时间
        # 开始读取排序后的数据
        for path in paths_sorted:
            # 先读取excel
            df = pd.read_excel(path, sheet_name=1)
            # Cycle_Index、Current(A)、Voltage(V)、Internal_Resistance(Ohm)
            # Step_Index、Test_Time(s)
            # 每一个cycle有9种状态，区分出每一个cycle
            # 其中，1是充电前的搁置状态、3表示恒流充电完成后的搁置状态，5为充电完成后的搁置状态、6是放电前的搁置状态8、9为放电后的搁置状态
            # 2是恒流充电、4是恒压充电、7是恒流放电
            # cls = set(df['Cycle_Index'])
            # cls = list(cls)
            cls = list(set(df['Cycle_Index']))  #set去重，list成组
            # 按每个循环来赋值
            for c in cls:
                df_c = df[df['Cycle_Index'] ==c ]  #第c个循环的表格

                # Charging
                df_ch=df_c[(df_c['Step_Index'] == 2) | (df_c['Step_Index'] == 4)]  # 取所有充电的表格
                ch_v = df_ch['Voltage(V)']  # 所有电压值
                ch_c = df_ch['Current(A)']  # 所有电流值
                ch_t = df_ch['Test_Time(s)']
                df_cc = df_c[df_c['Step_Index'] == 2]  # 恒流充电表格
                df_cv = df_c[df_c['Step_Index'] == 4]  # 恒压充电表格
                CCCT.append(np.max(df_cc['Test_Time(s)'])-np.min(df_cc['Test_Time(s)']))  # 每个循环恒流充电的总时间
                CVCT.append(np.max(df_cv['Test_Time(s)'])-np.min(df_cv['Test_Time(s)']))  # 每个循环恒压充电的总时间

                # Discharging
                df_dc = df_c[df_c['Step_Index'] == 7]
                dc_v = df_dc['Voltage(V)']
                dc_c = df_dc['Current(A)']
                dc_t = df_dc['Test_Time(s)']
                dc_ir = df_dc['Internal_Resistance(Ohm)']  # internal resistance
                # 计算放电量
                if len(list(dc_c)) != 0:
                    dc_t_diff = np.diff(list(dc_t))  # 取得放电间隔时间
                    dc_c = np.array(dc_c)[1:]  # 从第二个点开始对应第一个时间间隔
                    dc_capacity = np.abs(dc_c * dc_t_diff/3600)  # Q=A·h 电流×时间(s)/3600 1对1 电流是负值
                    # dc_capacity = np.sum(dc_capacity[:]) #单个循环的放电量之和
                    dc_capacity = [np.sum(dc_capacity[:n]) for n in range(dc_capacity.shape[0])]  # 每个循环中每个时间点的放电量
                    dc_capacities.append(dc_capacity[-1])
                    # soh计算 不是很清楚这个计算方式
                    dec = np.abs(np.array(dc_v)-3.8)[1:]  # np.argmin(dec)取得距离3.8V最近的一个点的序号
                    start = np.array(dc_capacity)[np.argmin(dec)]
                    dec = np.abs(np.array(dc_v)-3.4)[1:]
                    end = np.array(dc_capacity)[np.argmin(dec)]
                    state_of_health.append(end-start)  # 这边对soh的定义是从3.8V至3.4V的放电量定义的，之后可以重新定义一下
                    internal_resistance.append(np.mean(np.array(dc_ir)))
                    count += 1
                # list → array
                # pdb.set_trace()
        dc_capacities = np.array(dc_capacities)
        state_of_health = np.array(state_of_health)
        internal_resistance = np.array(internal_resistance)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)
        # pdb.set_trace()
        # print(dc_capacities)
        # 去除放电量异常点,以40个循环为一组，返回正常数据点的索引
        idx = drop_outliers(dc_capacities, count, 40)
        # 输出Battery[batt]
        df_tmp= pd.DataFrame({'cycle':np.linspace(1,idx.shape[0], idx.shape[0]),  # cls 从1-idx
                            'capacity':dc_capacities[idx],
                            'SoH':state_of_health[idx],
                            'resistance':internal_resistance[idx],
                            'CCCT':CCCT[idx],
                            'CVCT':CVCT[idx]})
        Battery[batt] = df_tmp
    return Battery


Battery = np.load('datasets/CALCE/CALCE.npy', allow_pickle=True)
Battery = Battery.item()
def plot_capacities(Battery_List):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b:', 'g--', 'r-.', 'c:']
    for batt,color in zip(Battery_List, color_list):
        battery = Battery[batt]
        ax.plot(battery['cycle'], battery['capacity'], color, label='Battery_'+batt)
    plt.plot([-1,1000],[Rated_Capacity*0.7, Rated_Capacity*0.7], c='black', lw=1, ls='--')  # 临界点直线
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 1°C')
    plt.legend()
    plt.show()

def plot_resistance(Battery_List):
    battery = Battery['CS2_35']
    plt.figure(figsize=(9, 6))
    plt.scatter(battery['cycle'], battery['SoH'], c=battery['resistance'], s=10)
    # c是颜色，而resistance是一个递增的array，默认是蓝色##s为散点面积 ##marker默认⭕
    cbar = plt.colorbar()
    cbar.set_label('Internal Resistance (Ohm)', fontsize=14, rotation=-90, labelpad=20)
    plt.xlabel('Number of Cycles', fontsize=14)
    plt.ylabel('State of Health', fontsize=14)
    plt.show()
def plot_others(Battery_List):
    battery = Battery['CS2_35']
    plt.figure(figsize=(12,9))
    names = ['capacity', 'resistance', 'CCCT', 'CVCT']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(battery['cycle'], battery[names[i]], s=10)
        plt.xlabel('Number of Cycles', fontsize=14)
        plt.ylabel(names[i], fontsize=14)
        plt.show()
if __name__ == '__main__':
    plot_capacities(Battery_List)
    plot_resistance(Battery_List)
    plot_others(Battery_List)
