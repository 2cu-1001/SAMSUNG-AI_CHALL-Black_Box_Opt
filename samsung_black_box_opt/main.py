import torch
import torch.nn as nn
import torch.optim as optim
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
        self.layer2 = nn.Linear(8, 4)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(4, 2)
        self.dropout3 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.dropout1(x)
        x = self.sigmoid(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.layer3(x))
        x = self.dropout3(x)
        return self.sigmoid(self.output_layer(x))

# def get_data():
#
# def preprocessing():


def main():
    start_time = time.time()
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    pre_file_path = "real_submission.csv"
    if os.path.isfile(pre_file_path):
        os.remove(pre_file_path)
    tmp_pred_file_path = "tmp_pred.txt"
    if os.path.isfile(tmp_pred_file_path):
        os.remove(tmp_pred_file_path)

    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    print("load data : done")

    drop_data = ['x_0', 'x_1', 'x_2', 'x_3']
    train_data = train_data.drop(drop_data, axis=1)
    test_data = test_data.drop(drop_data, axis=1)

    outlier = train_data[(abs((train_data['y'] - train_data['y'].mean()) / train_data['y'].std())) > 1.96].index
    train_data = train_data.drop(outlier)

    pre_X = train_data.values[36000:37000, 1:-1]
    pre_y = train_data.values[36000:37000, -1]
    pre_Xt = test_data.values[:, 1:]

    scaler = StandardScaler()
    pre_X = scaler.fit_transform(pre_X)
    pre_Xt = scaler.transform(pre_Xt)

    X, y, Xt = [], [], []

    train_sz = len(pre_X)
    test_sz = len(pre_Xt)
    dim = len(pre_X[0])

    for i in range(train_sz):
        if i % 100 == 0:
            print(i)
        for j in range(i + 1, train_sz):
            new_X = [a - b for a, b in zip(pre_X[i], pre_X[j])]
            new_y = 1 if pre_y[i] > pre_y[j] else 0
            X.append(new_X)
            y.append(new_y)

    print("generate new train data : done")
    print(time.time() - start_time)

    for i in range(test_sz):
        if i % 100 == 0:
            print(i)
        for j in range(test_sz):
            if i == j:
                continue
            new_Xt = [a - b for a, b in zip(pre_Xt[i], pre_Xt[j])]
            Xt.append(new_Xt)

    print("generate new test data : done")
    print(time.time() - start_time)

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(1)
    Xt = torch.FloatTensor(Xt)

    model = Model(dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)
    X, y, Xt, = X.to(device), y.to(device), Xt.to(device)
    print(next(model.parameters()).device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("initialize model : done")

    for epoch in range(1, 1001):
        output = model(X)
        cost = criterion(output, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch : {epoch}, Model : {list(model.parameters())}, Cost : {cost}")

    print("train data : done")
    print(list(model.parameters()))

    tmp_pred = model(Xt).squeeze(1)
    final_pred = torch.FloatTensor([0 for i in range(test_sz)]).to(device)
    print(tmp_pred.size())
    print(final_pred.size())

    tmp_pred_file = open("./tmp_pred.txt", "w")
    cnt = 0
    for cur_pred in tmp_pred:
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
        tmp_pred_file.write(f"{cur_pred}\n")


    print("predict and make submission file : done")


if __name__ == "__main__":
    main()