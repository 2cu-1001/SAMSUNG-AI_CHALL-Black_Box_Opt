import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import os
import time
from copy import deepcopy


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(dim, 8)
        self.dropout1 = nn.Dropout(0.5)
        # self.layer2 = nn.Linear(8, 4)
        # self.dropout2 = nn.Dropout(0.5)
        # self.layer3 = nn.Linear(4, 2)
        # self.dropout3 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

        self._initialize_weights()
    
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    init.ones_(module.bias)
    
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        # x = self.sigmoid(self.layer2(x))
        # x = self.dropout2(x)
        # x = self.sigmoid(self.layer3(x))
        # x = self.dropout3(x)
        return self.sigmoid(self.output_layer(x))


def load_data():
    pre_file_path = "real_submission.csv"
    if os.path.isfile(pre_file_path):
        os.remove(pre_file_path)
    tmp_pred_file_path = "./tmp_pred.txt"
    if os.path.isfile(tmp_pred_file_path):
        os.remove(tmp_pred_file_path)

    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    print("load data : done")

    return train_data, test_data


def preprocessing(train_data, test_data):
    drop_data = []
    train_data = train_data.drop(drop_data, axis=1)
    test_data = test_data.drop(drop_data, axis=1)
    
    for i in range(0, 11):
        cur_col = "x_" + str(i)
        cur_x_outlier = train_data[(abs((train_data[cur_col] - train_data[cur_col].mean()) / train_data[cur_col].std())) > 1.96].index
        train_data = train_data.drop(cur_x_outlier)
        print(f"x_{i} outlier drop : done")
    
    y_outlier = train_data[(abs((train_data['y'] - train_data['y'].mean()) / train_data['y'].std())) > 1.96].index
    train_data = train_data.drop(y_outlier)
    print("y outlier drop : done")

    pre_X = train_data.values[:, 1:-1]
    pre_y = train_data.values[:, -1]
    pre_Xt = test_data.values[:, 1:]

    scaler = StandardScaler()
    pre_X = scaler.fit_transform(pre_X)
    pre_Xt = scaler.transform(pre_Xt)

    # dim = 6
    # pca = PCA(n_components=dim)
    # pre_X = pca.fit_transform(pre_X)
    # pre_Xt = pca.transform(pre_Xt)

    X, y, Xt = [], [], []

    train_sz = len(pre_X)
    test_sz = len(pre_Xt)
    dim = len(pre_X[0])

    grid_sz = 700
    for k in range(0, train_sz - grid_sz, grid_sz):
        cur_st, cur_ed = k, k + grid_sz
        if k % 1000 == 0:
            print("grid :", k)

        for i in range(cur_st, cur_ed):
            if i % 100 == 0:
                print(i)
            for j in range(i + 1, cur_ed):
                new_X = [a - b for a, b in zip(pre_X[i], pre_X[j])]
                new_y = 1 if pre_y[i] > pre_y[j] else 0
                X.append(new_X)
                y.append(new_y)

    print("X size :", len(X))

    print("generate new train data : done")

    for i in range(test_sz):
        if i % 100 == 0:
            print(i)
        for j in range(test_sz):
            if i == j:
                continue
            new_Xt = [a - b for a, b in zip(pre_Xt[i], pre_Xt[j])]
            Xt.append(new_Xt)

    print("generate new test data : done")
    
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # Xt = scaler.transform(Xt)
    
    # dim = 4
    # pca = PCA(n_components=dim)
    # X = pca.fit_transform(X)
    # Xt = pca.transform(Xt)

    print("preprocessing : done")
    return X, y, Xt, dim, test_sz


def init_model(X, y, Xt, dim):
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(1)
    Xt = torch.FloatTensor(Xt)

    model = Model(dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)
    X, y, Xt, = X.to(device), y.to(device), Xt.to(device)
    print(next(model.parameters()).device)
    print(model.parameters())

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    print("initialize model : done")

    return X, y, Xt, model, criterion, optimizer, device


def training(X, y, Xt, model, criterion, optimizer, test_sz, device):
    for epoch in range(1, 10001):
        output = model(X)
        cost = criterion(output, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch : {epoch}, Model : {list(model.parameters())}, Cost : {cost}")
            
        if cost < 0.6:
            break

    print(f"Epoch : {epoch}, Model : {list(model.parameters())}, Cost : {cost}")
    print("train data : done")

    tmp_pred = model(Xt).squeeze(1).detach().cpu()
    tmp_pred_file = open("./tmp_pred.txt", "w")
    print("predict : done")

    cnt = 0
    for cur_pred in tmp_pred:
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt)
        tmp_pred_file.write(f"{cur_pred}\n")

    print("make submission file : done")


def main():
    start_time = time.time()
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    train_data, test_data = load_data()

    X, y, Xt, dim, test_sz = preprocessing(train_data, test_data)

    print("cur taken time :", time.time() - start_time)

    X, y, Xt, model, criterion, optimizer, device = init_model(X, y, Xt, dim)

    training(X, y, Xt, model, criterion, optimizer, test_sz, device)


if __name__ == "__main__":
    main()
