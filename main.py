#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Date: 2021.8.11
Author: Y. F. Zhang
"""

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
import numpy as np
import os

feature_dim = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def feature_extract(filename):
    data = pd.read_csv(filename,header=None).to_numpy(dtype=float)
    point_cloud = data[1:,0:3]
    ref = data[1:,3]
    [l,w,h,_] = data[0,:]
    max_x,min_x = np.max(point_cloud[:,0]),np.min(point_cloud[:,0])
    max_y, min_y = np.max(point_cloud[:,1]), np.min(point_cloud[:,1])
    # f1 = np.array([l,w,h])                          # 第一个特征
    f1 = np.array([h])   # 仅用h效果好些

    slices_num = 10
    if max_x-min_x > max_y-min_y:
        t = np.linspace(min_x,max_x,slices_num+1)   # slices_num+1个值构成slices_num个区间
        flag = 0
    else:
        t = np.linspace(min_y, max_y, slices_num+1)
        flag = 1
    f2 = []
    for i in range(len(t)-1):
        lb,ub = t[i],t[i+1]
        sum_h,num = 0,0
        for point in point_cloud:
            if lb < point[flag] < ub:
                sum_h += point[2]
                num += 1
        f2.append(sum_h/num if num!=0 else 0)
    f2 = np.array(f2)                                            # 第二个特征
    f3 = np.array([(max_x-min_x if flag==1 else max_y-min_y)/h]) # 第三个特征
    f4 = np.array([np.mean(ref),np.std(ref)])                    # 第四个特征
    num1,num2 = 0,0
    for r in ref:
        if 0<=r<0.2:
            num1 += 1
        if 0.2<=r<0.4:
            num2 += 1
    f5 = np.array([(num1 - num2) / point_cloud.shape[0]])  # 第五个特征
    feature = np.concatenate((f1,f2,f3,f4,f5))
    return feature

def load_data():
    data_car = np.array([[]],dtype=float).reshape(-1,feature_dim)
    for file in os.listdir('./training/car'):
        data_car = np.concatenate((data_car,feature_extract('./training/car/'+file).reshape(1,-1)),axis=0)
    label_car = np.zeros((data_car.shape[0],),dtype=float)   # car标记为0

    data_pedestrian = np.array([[]], dtype=float).reshape(-1, feature_dim)
    for file in os.listdir('./training/pedestrian'):
        data_pedestrian = np.concatenate((data_pedestrian, feature_extract('./training/pedestrian/' + file).reshape(1, -1)), axis=0)
    label_pederstian = 1+np.zeros((data_pedestrian.shape[0],), dtype=float)   # pedestrian标记为1

    data_cyclist = np.array([[]], dtype=float).reshape(-1, feature_dim)
    for file in os.listdir('./training/cyclist'):
        data_cyclist = np.concatenate((data_cyclist, feature_extract('./training/cyclist/' + file).reshape(1, -1)),axis=0)
    label_cyclist = 1+np.zeros((data_cyclist.shape[0],), dtype=float)  # cyclist标记为2

    data_others = np.array([[]], dtype=float).reshape(-1, feature_dim)
    for file in os.listdir('./training/others'):
        data_others = np.concatenate((data_others, feature_extract('./training/others/' + file).reshape(1, -1)),axis=0)
    label_others = 1 + np.zeros((data_others.shape[0],), dtype=float)  # others标记为3

    total_data = np.concatenate((data_car,data_pedestrian,data_cyclist,data_others),axis=0)
    total_label = np.concatenate((label_car,label_pederstian,label_cyclist,label_others))

    train_x,test_x,train_y,test_y = train_test_split(total_data,total_label)


    train_x = torch.from_numpy(train_x).type(torch.float32).to(device)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor).to(device)
    test_x = torch.from_numpy(test_x).type(torch.float32).to(device)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor).to(device)

    class Mydataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __getitem__(self, index):
            feature = self.features[index]
            label = self.labels[index]
            return feature, label

        def __len__(self):
            return len(self.features)

    train_ds = Mydataset(train_x, train_y)
    test_ds = Mydataset(test_x, test_y)
    BTACH_SIZE = 256
    train_dl = torch.utils.data.DataLoader(
                train_ds,
                batch_size=BTACH_SIZE,
                shuffle=True)
    test_dl = torch.utils.data.DataLoader(
                test_ds,
                batch_size=BTACH_SIZE,
                shuffle=True)
    print('=====load data finished!+=====')
    return train_dl,test_dl,(train_ds,test_ds)

def get_model():
    class Model(nn.Module):
        def __init__(self,dim):
            super().__init__()
            self.liner_1 = nn.Linear(dim,256)
            self.liner_2 = nn.Linear(256,256)
            self.liner_3 = nn.Linear(256,2)
            self.relu = nn.LeakyReLU()
        def forward(self,feature):
            x = self.liner_1(feature)
            x = self.relu(x)
            x = self.liner_2(x)
            x = self.relu(x)
            x = self.liner_3(x)
            return x

    model = Model(feature_dim).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    return model,opt,loss_fn

def save_model_para(model):
    import scipy.io as scio
    var_name = list()
    for name,para in model.named_parameters():
        x = name.split(".")
        para_name = x[0] + "_" + x[1]
        exec(para_name + '=para.cpu().data.numpy()')
        print(name)
        var_name.append(para.cpu().data.numpy())
    data_file = 'para_save_open.mat'
    scio.savemat(data_file,
                 {'l1_weight': var_name[0], 'l1_bias': var_name[1], 'l2_weight': var_name[2], 'l2_bias': var_name[3],
                  'l3_weight': var_name[4], 'l3_bias': var_name[5],
                  })
def test():
    model = get_model()[0]
    model.load_state_dict(torch.load('model_state_dict.pkl'))
    test_feature = np.array([[]], dtype=float).reshape(-1, feature_dim)
    test_data = list()
    for file in os.listdir('./testing/cluster'):
        data = pd.read_csv("".join(['./testing/cluster/',file]), header=None).to_numpy(dtype=float)
        test_data.append(data)
        feature = feature_extract("".join(['./testing/cluster/',file]))
        test_feature = np.concatenate((test_feature,feature.reshape(1,-1)),axis=0)
    test_x = torch.from_numpy(test_feature).type(torch.float32).to(device)

    model.eval()
    plt.figure(2)
    plt.scatter(0,0,s=20)
    with torch.no_grad():
        output = model(test_x)
        y_pred = torch.argmax(output,dim=1).cpu().detach().numpy()
        # points_dict = {'Car':np.array([]),'Pedestrian':np.array([]),'Cyclist':np.array([]),'Others':np.array([])}
        # category_list = ['Car','Pedestrian','Cyclist','Others']
        points_dict = {'Car':np.array([]),'Others':np.array([])}
        category_list = ['Car','Others']
        for i,val in enumerate(y_pred):
            points_dict[category_list[val]] = np.append(points_dict[category_list[val]],test_data[i][1:,0:3])
        for ca in category_list:
            points = points_dict[ca].reshape(-1,3)
            plt.scatter(points[:,0],points[:,1],s=np.ones(len(points),))
        leg = ['Coordinate origin of lidar']
        leg.extend(category_list)
        plt.legend(leg)
    plt.show()

def test2():
    model = get_model()[0]
    model.load_state_dict(torch.load('model_state_dict.pkl'))
    test_feature = np.array([[]], dtype=float).reshape(-1, feature_dim)
    test_data = list()
    for file in os.listdir('./training/pedestrian'):
        data = pd.read_csv("".join(['./training/pedestrian/', file]), header=None).to_numpy(dtype=float)
        test_data.append(data)
        feature = feature_extract("".join(['./training/pedestrian/', file]))
        test_feature = np.concatenate((test_feature, feature.reshape(1, -1)), axis=0)
    test_x = torch.from_numpy(test_feature).type(torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        output = model(test_x)
        y_pred = torch.argmax(output, dim=1).cpu().detach().numpy()
        print((y_pred==np.ones(len(y_pred),).mean()))

def main():
    train_dl,test_dl,(train_ds,test_ds) = load_data()
    model,opt,loss_fn = get_model()

    def accuracy(y_pred,y_true):
        y_pred = torch.argmax(y_pred,dim=1)
        acc = (y_pred == y_true).float().mean()
        return acc

    epochs = 400
    loss_list = []
    accuracy_list = []
    test_loss_list = []
    test_accuarcy_list = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            epoch_accuracy = accuracy(model(train_ds.features), train_ds.labels)
            epoch_loss = loss_fn(model(train_ds.features), train_ds.labels).data
            epoch_test_accuracy = accuracy(model(test_ds.features), test_ds.labels)
            epoch_test_loss = loss_fn(model(test_ds.features), test_ds.labels).data
            loss_list.append(round(epoch_loss.item(), 3))
            accuracy_list.append(round(epoch_accuracy.item(), 3))
            test_loss_list.append(round(epoch_test_loss.item(), 3))
            test_accuarcy_list.append(round(epoch_test_accuracy.item(), 3))
            print('epoch: ', epoch, 'loss： ', round(epoch_loss.item(), 3),
                  'accuracy:', round(epoch_accuracy.item(), 3),
                  'test_loss： ', round(epoch_test_loss.item(), 3),
                  'test_accuracy:', round(epoch_test_accuracy.item(), 3)
                  )
    save_model_para(model)
    torch.save(model.state_dict(),'model_state_dict.pkl')
    plt.figure(1)
    plt.plot(range(0, epochs), loss_list, 'r', label='Training loss')
    plt.plot(range(0, epochs), accuracy_list, 'g', label='Training accuracy')
    plt.plot(range(0, epochs), test_loss_list, 'b', label='Test loss')
    plt.plot(range(0, epochs), test_accuarcy_list, 'k', label='Test accuracy')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    def setup_seed(seed):
        """设置随机数种子函数"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(188)
    test()