# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy.random import *
import copy
import get_image as GI

#Config Class
class config():
    def __init__(self):
        self.max_update = 20000
        self.epoch = 10
        self.noise = 0.4
        self.memory_num = 4
        self.test_num = 0
        self.model = 0#0:Hopfield Network, 1:Associated memory model
        self.plot = 1 #0:nonplot,1:plot
        self.energy_plot = 1


#Hopfield Network & Associative memory model
class Model():
    def __init__(self, data, config):
        self.data = data[:config.memory_num]
        self.memory_num = config.memory_num
        self.data_size = int(data[0].shape[0])
        self.noise = config.noise
        self.max_update = config.max_update
        self.epoch = config.epoch
        self.W = np.zeros((self.data_size**2,self.data_size**2))
        self.model = config.model
        self.plot_flag = config.plot
        self.energy_flag = config.energy_plot
        self.energy_list = []
    # エネルギー計算の関数
    def energy(self, x):
        return -0.5*np.dot(x.T, np.dot(self.W, x))

    # 最大絶対誤差
    def Error_Rate(self,test_num, x):
        # 対象データと訓練データの差を計算
        return (np.abs(self.data[test_num] - x)/2).sum()/self.data_size**2*100

    # テストデータの作成
    def noise_make(self, test_idx):
        x_test = copy.deepcopy(self.data[test_idx])
        # 確率rateで符号を反転させる
        flip = choice([1, -1],self.data_size, p=[1 - self.noise, self.noise])
        x_test = x_test * flip

        return x_test

    # 学習(連想記憶モデルとホップフィールドで共通)
    def fit(self,u):
        datas = self.data[u].reshape(self.data_size**2,1)
        self.W += np.dot(datas, datas.T)
        for i in range(self.data_size**2):
            self.W[i, i] = 0
        return self.W

    # ホップフィールドネットワークの想起
    def HN(self, test_data):
        self.energy_list = []
        for _ in range(self.max_update):
            #テストデータの更新（非同期更新）
            num = randint(self.data_size**2)
            test_data = test_data.reshape(self.data_size**2,1)
            test_data[num] = np.sign(np.dot(self.W[num], test_data))
            if(self.energy_flag == 1):
                self.energy_list.append(self.energy(test_data)[0][0])
        test_data = test_data.reshape(self.data_size,self.data_size)

        return test_data

    # 連想記憶モデルの想起
    def AMM(self, test_data):
        self.energy_list = []
        test_data = test_data.reshape(self.data_size**2,1)
        for _ in range(self.max_update):
            # テストデータの更新(同期更新)
            test_data = np.sign(np.dot(self.W, test_data))
            if(self.energy_flag == 1):
                self.energy_list.append(self.energy(test_data)[0][0])
        test_data = test_data.reshape(self.data_size,self.data_size)
        return test_data
    #画像プロット
    def plot(self, datas, name='example'):
        plt.imshow(datas, cmap='Oranges')
        plt.title = name
        plt.show()

    def energy_plot(self):
        plt.plot(range(1,len(self.energy_list)+1),self.energy_list)
        plt.show()    
    def run(self, test_idx):
        err= 0  # 正解と異なるマスの数
        acc = 0  # 正解率

        # 訓練データから重み行列の計算
        for i in range(self.memory_num):
            self.W = self.fit(i)

        for l in tqdm(range(self.epoch)):
            # テストデータの作成
            test_data = self.noise_make(test_idx)
            if(self.plot_flag == 1):
                self.plot(test_data,"Noised_Picture")
            # テストデータからの想起
            test_predict = self.AMM(test_data) if self.model else self.HN(test_data)
            if(self.plot_flag == 1):
                self.plot(test_predict,"Test")
            
            # 正答率，誤差率の計算
            _err = self.Error_Rate(test_idx,test_predict)
            err += _err
            if _err == 0:
                acc += 1

        if(self.energy_flag == 1):
            self.energy_plot()
        err/= self.epoch
        acc /= float(self.epoch)
        print("error_rate = {0}".format(err))
        print("accuracy = {0}".format(acc))


def main():
    configs = config()
    data = list([])
    path = 'data/Lenna.bmp'
    data.append(GI.get_images(path).T.astype(np.float32))
    path = 'data/Mandrill.bmp'
    data.append(GI.get_images(path).T.astype(np.float32))
    path = 'data/Pepper.bmp'
    data.append(GI.get_images(path).T.astype(np.float32))
    path = 'data/Parrots.bmp'
    data.append(GI.get_images(path).T.astype(np.float32))
    hop = Model(data, configs)
    hop.run(configs.test_num)


if __name__ == '__main__':
    main()