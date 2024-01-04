# -*- coding: utf-8 -*-

import sys
import os
import random
import math
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

import torch.nn.functional as F
#import torchvision
import pdb


# import CACLE_Load as load
# 不止要把设置里面的解释器改成自己创建的环境，在terminal中也要conda activate
# 所有plt加上plt.show()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# print(torch.cuda.is_available())
Battery_List = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
dir = './dataset'
Rated_Capacity = 1.1
# Battery = {}
# print(load.drop_outliers(np.arange(10),10,2))
# Battery=load.load_data(Battery_List,dir)
# print(Battery)
# pdb.set_trace()


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size] # [1,1+win], [2,2+win]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)

def get_train_test(data_dict, batt, window_size=8):
    data_sequence = data_dict[batt]['capacity']
    train_data, test_data = data_sequence[:window_size + 1], data_sequence[window_size + 1:]
    # 当前batt的前第0个到第win个值作为train_data，往后的作为test_data
    train_x, train_y = build_sequences(text = train_data, window_size = window_size)
    # train_x = [0:win]  train_y = [1:1+win]
    for k, v in data_dict.items():
        # k 为4个batt
        # v 为字典
        if k != batt: # 另外三个batt作为training的数据集
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
            # data_x = [[0:win], [1:1+win],...,[len-win-1,len(text)-1]]
            # data_y = [[1:1+win], [2:2+win],...,[len-win-1,len(text)-1],[len-win,len(text)]]
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            # train_x是由当前batt的train_x和另外三个batt的data_x按纵向一行行排起来的
    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re) / true_re if abs(true_re - pred_re) / true_re <= 1 else 1

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = math.sqrt(mean_squared_error(y_test, y_predict))
    return mae, rmse

def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed)  # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

### 读取信息
# Battery_List=['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
# dir = './dataset'
# dir='./batt/CALCE/dataset/'
# dir = '.../batt/CALCE/CALCE/dataset'
# dir = 'E:/DL/1workspace/batt/CALCE/CALCE/dataset'



#数据读取
Battery=np.load('datasets/CALCE/CALCE.npy',allow_pickle=True).item()

#模型单元设计
class Net(nn.Module):
    def __init__(self,input_size,hidden_size, num_layers,n_class=1,mode="LSTM"):

        super(Net,self).__init__()
        #隐藏层尺寸
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.5)
        #神经网络训练单元
        self.cell = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        if mode == "GRU":
            self.cell = nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        elif mode == "RNN":
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self,x):
        output,_ = self.cell(x) # output就是[batchsize,seqlen,hiddensize]
        output = output.reshape(-1,self.hidden_size) # output改成 [-1,hiddensize]
        output = self.linear(output) # [-1,1]
        return output


def train(lr=1e-2,input_size=16,hidden_size=128,num_layers=2,weight_decay=0,mode='LSTM',EPOCH=1000,seed=0):
    score_list = []
    result_list = []
    for i in range(4):

        #获取电池编号
        batt = Battery_List[i]

        #得到训练集和测试集
        train_x, train_y, train_data, test_data = get_train_test(Battery, batt, window_size=input_size)
        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        setup_seed(seed)
        model = Net(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, mode=mode)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        criterion.to(device)
        test_x = train_data.copy()
        loss_list, y_ = [0], []
        mae, rmse, re = 1, 1, 1
        score_, score = 1, 1
        y_result = []
        img_num = 0

        #模型训练
        for epoch in range(EPOCH):

            #训练过程处理
            X = np.reshape(train_x / Rated_Capacity, (-1, 1, input_size)).astype(np.float32)  # (batch_size, seq_len, input_size)
            # batch_size 表示几个时间单位的数据
            # seq_len 表示时间单位内的数据数
            # input_size 表示喂的数据的长度
            y = np.reshape(train_y[:, -1] / Rated_Capacity, (-1, 1)).astype(np.float32)  # shape 为 (batch_size, 1)
            # y 取的是最后序列的最后一个值，即每个X sequence的后一个点
            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            output = model(X) # （batch_size,seq_len, hidden_size*num_direction) num_direction是双向传播就为2
            output = output.reshape(-1, 1) # 展开
            loss = criterion(output, y) # 每个模型预测值y和y_hat算loss然后bachpropagation
            # 清零gradient,loss BP ,使用gradient优化
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if (epoch + 1) % 100 == 0:# 第99个和199 299399 999epoch的时候预测
                test_x = train_data.copy()  # 每100次重新预测一次 训练100个epoch就预测一次当前batt的曲线
                # train_data [0:inputsize+1]
                # test_x [-inputsize:]
                #train_data[input_size] = test_x[-1]
                #test_x[-inputsize] = train_data[1]

                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data): # text_x一直在加点，然后直到最后一次预测时len(text_x)-len(test_data) 等于len(test_data) 就预测完成
                    x = np.reshape(np.array(test_x[-input_size:]) / Rated_Capacity, (-1, 1, input_size)).astype(
                        np.float32)
                    x = torch.from_numpy(x).to(device)  # shape: (batch_size, 1, input_size)
                    pred = model(x) # pred shape 为 [-1,1] -1为自适应值
                    next_point = pred.data.cpu().numpy()[0, 0] * Rated_Capacity # 取二维数组中的第一个点，即为预测点

                    test_x.append(next_point)  # 测试值加入原来序列用来继续预测下一个点
                    point_list.append(next_point)  # 保存输出序列最后一个点的预测值


                y_ = list(y_)
                y_.append(point_list)  # 保存本次预测【所有的】预测值
                loss_list.append(loss)
                # print('y_ is:',y_)
                # x_ = np.linspace(1,763,763) # [1:763]  float
                # plot_pred_capacities(x_,np.array(y_)[0])
                # pdb.set_trace()
                # y_ = np.array(y_) # 1*763 2D
                # print('y_ shape is:',y_.shape)
                # pdb.set_trace()
                mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity * 0.7)
                print('epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mae,rmse, re))
                # plot_pred_capacities(Battery,y_) # 画predict曲线
                # y_result.append(y_)
                # print('y_result is:', y_result)

                # y_result = np.reshape(-1,9)
                # print('y_result\'s  new shape is:', np.array(y_result).shape)
                # pdb.set_trace()
                #
                # for i in range(10):
                #     color = ['b-', 'r--']
                #     fig, ax = plt.subplots(2, 5, i + 1, figsize=(3, 2))
                #     ax.plot(Battery_List[batt]['cls'], Battery_List[batt]['capacities'], color[0], label='Battery_' + batt)
                #     ax.plot(Battery_List[batt]['cls'][input_size + 1:], y_, color[1])
                #     plt.legend()
                #     plt.show()
                img_num += 1
            score = [re, mae, rmse]

            if (loss < 1e-3) and (score_[0] < score[0]): #loss小于 0.001 且 前一次的相对误差RE比后一次的小  停
                break

            #记录前一次的预测效果，用作与下一次进行比较
            score_ = score.copy()


        # #按照训练方案找到最好的模型以及模型分数之后，将结果进行可视化展示
        # print('cls\'s length is:', len(Battery[batt]['cls']))
        # # print('capacities:', Battery[batt]['capacities'])
        # print('capacities length is :', len(Battery[batt]['capacities']))
        # print('img_number is :',img_num)
        # y_result = np.array(y_result)
        # print('y_result is:',y_result)
        # 绘图，10次训练图像
        # pdb.set_trace()//////////////////////////////////////////////////////////////////////////////////////pop
        # if img_num % 2 == 0:
        #     # fig, axes = plt.subplots(2, img_num/2)//////////////////////////////////////////////////////////pop
        #     fig, axes = plt.subplots(2, 4)
        # else:
        #     fig,axes = plt.subplots(2,int((img_num+1)/2)) # 不能是float
        # axes_list = []
        # color = ['b', 'r']
        # y_index = 0
        # cls_idx = Battery[batt]['cls']
        #
        # # cls_idx_keys = Battery[batt]['cls'].keys()
        # # cls_idx_values
        # print(type(cls_idx))
        # # pdb.set_trace()
        # cls_idx = list(cls_idx)
        #
        # print('cls_idx:',cls_idx)
        # y_ = np.array(y_)
        # print('y_shape',y_.shape)
        #
        # # print('y_result is:',y_result)
        # for i in range(axes.shape[0]):
        #     for j in range(axes.shape[1]):
        #         axes_list.append(axes[i,j])
        # for ax in axes_list:
        #     if y_index == img_num:
        #         break
        #     plt.xlim(0,1000)
        #     plt.ylim(0.0,1.2)
        #     y_plot = y_[y_index].T
        #     y_plot = list(y_plot)
        #     # for i in range(len(y_result[y_index])):
        #     # for i in range(len(y_[y_index])):
        #     #     y_plot.append(y_[y_index][i])
        #     # print('y_plot',y_plot)
        #     # pdb.set_trace()
        #     ax.plot(Battery[batt]['cls'], Battery[batt]['capacities'], color=color[0], label='Battery_' + batt + 'real') # 第一条线
        #     ax.plot(cls_idx[input_size + 1:], y_plot, color=color[1], label='Battery_' + batt + 'pred')
        #     # plt.legend(loc='upper right')  # 绘制图例，指定图例位置
        #     plt.legend()
        #     # ax.plot(cls_idx[input_size+1:], y_plot[0], color=color[1],label='Battery_' + batt + 'pred') # 第二条曲线
        #     y_index += 1
        #     plt.xticks([])  # 去掉x轴刻度
        # plt.show()
        score_list.append(score_)
        result_list.append(y_[-1])


    return score_list, result_list
if __name__ == '__main__':
    window_size = 128
    EPOCH = 1000
    lr = 0.001    # learning rate  0.01 epoch 10
    hidden_size = 256
    num_layers = 2
    weight_decay = 0.0
    mode = 'LSTM'# RNN, LSTM, GRU
    Rated_Capacity = 1.1

    SCORE = []

    for seed in range(10):
        print('seed: ', seed)
        score_list, _ = train(lr=lr, input_size=window_size, hidden_size=hidden_size, num_layers=num_layers,weight_decay=weight_decay, mode=mode, EPOCH=EPOCH, seed=seed)
        print('------------------------------------------------------------------')
        for s in score_list:
            SCORE.append(s)

    mlist = ['re', 'mae', 'rmse']
    for i in range(3):
        s = [line[i] for line in SCORE]
        print(mlist[i] + ' mean: {:<6.4f}'.format(np.mean(np.array(s))))