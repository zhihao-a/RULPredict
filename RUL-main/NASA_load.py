#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io
from datetime import datetime
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')





# 转换时间格式，将字符串转换成 datatime 格式
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# 加载 mat 文件
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data


# 提取锂电池容量
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]


# 获取锂电池充电或放电时的测试数据
def getBatteryValues(Battery, Type='charge'):
    data=[]
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data




if __name__ == '__main__':

    Battery_list = ['B0005', 'B0006', 'B0007', 'B0018'] # 4 个数据集的名字
    dir_path = 'datasets/NASA/'

    capacity, charge, discharge = {}, {}, {}
    # battery={}
    # for name in Battery_list:
    #     print('Load Dataset ' + name + '.mat ...')
    #     path = dir_path + name + '.mat'
    #     data = loadMat(path)
    #     battery[name] = data
    #     capacity[name] = getBatteryCapacity(data)              # 放电时的容量数据
    #     charge[name] = getBatteryValues(data, 'charge')        # 充电数据
    #     discharge[name] = getBatteryValues(data, 'discharge')  # 放电数据
    # np.save('datasets/NASA/NASA.npy',battery)


    ### 如果上面的数据集读取失败，可以通过下面的方式加载已提取出来的数据

    Battery = np.load('datasets/NASA/NASA.npy', allow_pickle=True)
    Battery = Battery.item()
    for name in Battery_list:
        print('Load Dataset ' + name + '.mat ...')
        data = Battery[name]
        capacity[name] = getBatteryCapacity(data)              # 放电时的容量数据
        charge[name] = getBatteryValues(data, 'charge')        # 充电数据
        discharge[name] = getBatteryValues(data, 'discharge')  # 放电数据

    ### 3. 容量 v.s. 充放电次数 曲线

    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b:', 'g--', 'r-.', 'c.']
    c = 0
    for name in Battery_list:
        data = capacity[name]
        color = color_list[c]
        ax.plot(data[0], data[1], color, label=name)
        c += 1
    plt.plot([-1,170],[2.0*0.7,2.0*0.7],c='black',lw=1,ls='--')  # 临界点直线
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
    plt.legend()



    # ### 4. 充电电流 v.s. 充电时间 曲线

    name = 'B0005'       #查看的电池号
    time = [0, 50, 100] #查看的充电次数

    # 画图
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b', 'g', 'r']
    c = 0
    for t in time:
        Battery = charge[name][t]
        color = color_list[c]
        ax.plot(Battery['Time'], Battery['Current_measured'], color, label='charge time: '+str(t))
        c += 1
    ax.set(xlabel='Time', ylabel='Current (A)', title='Charging Curve')
    plt.legend()


    # ### 5. 放电电压 v.s. 充电时间 曲线

    name = 'B0005'          #查看的电池号
    time = [0, 50, 100]     #查看的放电次数

    # 画图
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b', 'g', 'r']
    c = 0
    for t in time:
        Battery = discharge[name][t]
        color = color_list[c]
        ax.plot(Battery['Time'], Battery['Voltage_measured'], color, label='discharge time: '+str(t))
        c += 1
    ax.set(xlabel='Time', ylabel='Voltage (V)', title='Discharging Curve')
    plt.legend()
    plt.show()








