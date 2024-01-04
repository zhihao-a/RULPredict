#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import random
import math
import time
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

# get_ipython().run_line_magic('matplotlib', 'inline')

from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

Rated_Capacity = 1.1

def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, window_size=8):
    data_sequence=data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re if abs(true_re - pred_re)/true_re<=1 else 1


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae,rmse
    
    
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('datasets/CALCE/CALCE.npy', allow_pickle=True)
Battery = Battery.item()


# # Rated_Capacity = 1.1
# fig, ax = plt.subplots(1, figsize=(12, 8))
# color_list = ['b:', 'g--', 'r-.', 'c.']
# for name,color in zip(Battery_list, color_list):
#     df_result = Battery[name]
#     ax.plot(df_result['cycle'], df_result['capacity'], color, label='Battery_'+name)
# ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 1°C')
# plt.legend()
# plt.show()





class Autoencoder(nn.Module):
    def __init__(self, input_size=16, hidden_dim=8, noise_level=0.001):
        super(Autoencoder, self).__init__()
        self.input_size, self.hidden_dim, self.noise_level = input_size, hidden_dim, noise_level
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_size)
        
    def encoder(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1
    
    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x
    
    def decoder(self, x):
        h2 = self.fc2(x)
        return h2
    
    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=16):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x    
    
    
class Net(nn.Module):
    def __init__(self, feature_size=16, hidden_dim=32, num_layers=1, nhead=8, dropout=0.0, noise_level=0.01):
        super(Net, self).__init__()
        self.auto_hidden = int(feature_size/2)
        input_size = self.auto_hidden 
        self.pos = PositionalEncoding(d_model=input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear1 = nn.Linear(input_size, 1)
        self.linear2 = nn.Linear(int(input_size/2), 1)
        self.autoencoder = Autoencoder(input_size=feature_size, hidden_dim=self.auto_hidden, noise_level=noise_level)
 
    def forward(self, x): 
        batch_size, feature_num, feature_size  = x.shape 
        encode, decode = self.autoencoder(x.reshape(batch_size, -1))# batch_size*seq_len
        out = encode.reshape(batch_size, -1, self.auto_hidden)
        out = self.pos(out)
        out = out.reshape(1, batch_size, -1) # (1, batch_size, input_size)
        out = self.cell(out)                 # shape (1, batch_size, input_size)
        out = out.reshape(batch_size, -1)    # (batch_size, input_size)
        out = self.linear1(out)              # (batch_size, 1)
        
        return out, decode





def train(lr=0.01, feature_size=8, hidden_dim=32, num_layers=1, nhead=8, weight_decay=0.0, EPOCH=1000, seed=0, 
         alpha=0.0, noise_level=0.0, dropout=0.0, metric='re', is_load_weights=True):
    score_list, result_list = [], []
    
    for i in range(4):
        name = Battery_list[i]
        window_size = feature_size
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size) 
        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        setup_seed(seed)
        model = Net(feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout,
                    noise_level=noise_level)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss().to(device)

        '''
        # save ramdom data for repetition
        if torch.__version__.split('+')[0] >= '1.6.0':
            torch.save(model.state_dict(), 'model_CALCE'+str(seed)+'.pth')
        else:
            torch.save(model.state_dict(), 'model_CALCE.pth', _use_new_zipfile_serialization=False)
         '''

        # load the random data generated by my device
        if is_load_weights: 
            if torch.__version__.split('+')[0] >= '1.6.0':
                model.load_state_dict(torch.load('initial_weights/model_CALCE.pth')) 
            else:
                model.load_state_dict(torch.load('initial_weights/model_CALCE_1.5.0.pth'))
                
        test_x = train_data.copy()
        loss_list, y_ = [0], []
        rmse, re,mae = 1, 1, 1
        score_, score = [1], [1]

        for epoch in range(EPOCH):
            X = np.reshape(train_x/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)#(batch_size, seq_len, input_size)
            y = np.reshape(train_y[:,-1]/Rated_Capacity,(-1,1)).astype(np.float32)# shape 为 (batch_size, 1)

            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            output, decode = model(X)
            output = output.reshape(-1, 1)

            #共同训练 去噪和预测 两部分，提出的目标函数
            loss = criterion(output, y) + alpha * criterion(decode, X.reshape(-1, feature_size))
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            if (epoch + 1)%10 == 0:
                test_x = train_data.copy()
                point_list = []

                #不断地预测之后的容量值，直到预测完毕
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size:])/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device) # (batch_size,feature_size=1,input_size)
                    pred, _ = model(x)                 # pred shape (batch_size=1, feature_size=1)
                    next_point = pred.data.cpu().numpy()[0,0] * Rated_Capacity
                    test_x.append(next_point)     # The test values are added to the original sequence to continue to predict the next point
                    point_list.append(next_point) # Saves the predicted value of the last point in the output sequence


                y_.append(point_list)             # Save all the predicted values
                loss_list.append(loss)
                mae,rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity*0.7)
                #print('epoch:{:<2d} | loss:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, rmse, re))
            if metric == 're':
                score = [re]
            elif metric== 'mae':
                score  [mae]
            elif metric == 'rmse':
                score = [rmse]
            else:
                score = [re, mae,rmse]
            if (loss < 0.001) and (score_[0] < score[0]):
                break
            score_ = score.copy()
            
        score_list.append(score_)
        result_list.append(y_[-1])
    return score_list, result_list
# ### use grid-search to determ parameters
def gridSearch():
    Rated_Capacity = 1.1
    window_size = 64
    feature_size = window_size
    dropout = 0.0
    EPOCH = 500
    nhead = 16
    is_load_weights = False
    weight_decay = 0.0
    noise_level = 0.0
    num_layers = 1
    metric = 're'

    states = {}
    for lr in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        for hidden_dim in [16, 32, 64]:
            for num_layers in [1,2]:
                for alpha in [1e-4, 1e-3, 1e-2]:
                    show_str = 'lr={}, num_layers={}, hidden_dim={},alpha={}'.format(lr, num_layers, hidden_dim,alpha)
                    print(show_str)
                    SCORE = []
                    for seed in range(5):
                        print('seed:{}'.format(seed))
                        score_list, _ = train(lr=lr, feature_size=feature_size, hidden_dim=hidden_dim,
                                              num_layers=num_layers, nhead=nhead,
                                              weight_decay=weight_decay, EPOCH=EPOCH, seed=seed, dropout=dropout,
                                              alpha=alpha,
                                              noise_level=noise_level, metric=metric, is_load_weights=is_load_weights)
                        print(np.array(score_list))
                        print(metric + ': {:<6.4f}'.format(np.mean(np.array(score_list))))
                        print('------------------------------------------------------------------')
                        for s in score_list:
                            SCORE.append(s)

                    print(metric + ' mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))
                    states[show_str] = np.mean(np.array(SCORE))
                    print('===================================================================')

        min_key = min(states, key=states.get)
        print('optimal parameters: {}, result: {}'.format(min_key, states[min_key]))

# ### optimal parameters of my device : Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz, Win10
# 
# Rated_Capacity = 1.1
# window_size = 64
# feature_size = window_size
# dropout = 0.0
# EPOCH = 500
# nhead = 16
# weight_decay = 0.0
# noise_level = 0.0
# alpha = 0.01
# lr = 0.0005    # learning rate
# hidden_dim = 32
# num_layers = 1
# is_load_weights = False
# metric = 're'
# re mean: 0.0536
# 
# Rated_Capacity = 1.1
# window_size = 64
# feature_size = window_size
# dropout = 0.0
# EPOCH = 500
# nhead = 16
# weight_decay = 0.0
# noise_level = 0.0
# alpha = 0.01
# lr = 0.0005    # learning rate
# hidden_dim = 32
# num_layers = 1
# is_load_weights = False
# metric = 'rmse'
# rmse mean: 0.0690



if __name__ == '__main__':
    # gridSearch()
    Rated_Capacity = 1.1
    window_size = 64
    feature_size = window_size
    dropout = 0.0
    EPOCH = 500
    nhead = 16
    weight_decay = 0.0
    noise_level = 0.0
    alpha = 0.01
    lr = 0.0005    # learning rate
    hidden_dim = 32
    num_layers = 1
    is_load_weights = True
    metric = 're'

    seed = 0
    SCORE = []
    print('seed:{}'.format(seed))
    score_list, _ = train(lr=lr, feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead,
                          weight_decay=weight_decay, EPOCH=EPOCH, seed=seed, dropout=dropout, alpha=alpha,
                          noise_level=noise_level, metric=metric, is_load_weights=is_load_weights)
    print(np.array(score_list))
    print(metric + ': {:<6.4f}'.format(np.mean(np.array(score_list))))
    print('------------------------------------------------------------------')
    for s in score_list:
        SCORE.append(s)

    print(metric + ' mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))









