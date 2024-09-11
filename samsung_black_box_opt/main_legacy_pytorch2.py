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
        self.layer1 = nn.Linear(dim, 1)
        self.bn1 = nn.BatchNorm1d(1)
        # self.dropout1 = nn.Dropout(0.5)
        # self.layer2 = nn.Linear(6, 3)
        # self.bn2 = nn.BatchNorm1d(3)
        # self.dropout2 = nn.Dropout(0.5)
        # self.layer3 = nn.Linear(3, 2)
        # self.bn3 = nn.BatchNorm1d(2)
        # self.dropout3 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(1, 1)
        self.relu = nn.ReLU()
        
        self._initialize_weights()
    
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight, gain=5.0)
                if module.bias is not None:
                    init.ones_(module.bias)
                    

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.bn1(x)
        # x = self.dropout1(x)
        # x = self.relu(self.layer2(x))
        # x = self.bn2(x)
        # x = self.dropout2(x)
        # x = self.relu(self.layer3(x))
        # x = self.bn3(x)
        # x = self.dropout3(x)
        return self.output_layer(x)


def get_data():
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    
    x_outlier = []
    for i in range(0, 11):
        cur_col = "x_" + str(i)
        cur_x_outlier = train_data[(abs((train_data[cur_col] - train_data[cur_col].mean()) / train_data[cur_col].std())) > 1.96].index
        x_outlier = list(set(x_outlier) | set(cur_x_outlier))
        # train_data = train_data.drop(cur_x_outlier)
        # print(f"x_{i} outlier drop : done")
    train_data = train_data.drop(x_outlier)    
     
    # train_data = train_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)
    # test_data = test_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)
    
    # sns.pairplot(train_data)
    # plt.show()
    # print(len(train_data))
    # plt.scatter([i for i in range(len(train_data))], train_data['y'], s=2)
    outlier = train_data[(abs((train_data['y'] - train_data['y'].mean()) / train_data['y'].std())) > 1.645].index
    train_data = train_data.drop(outlier)
    # plt.scatter([i for i in range(len(train_data))], train_data['y'], s=2, alpha=0.8)
    # plt.show()

    # train_data = train_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)
    # test_data = test_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)

    X = train_data.values[:, 1:-1]
    y = train_data.values[:, -1]
    Xt = test_data.values[:, 1:]

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # Xt = scaler.transform(Xt)

    dim = 11

    # dim = 4
    # lle = LocallyLinearEmbedding(n_components=dim)
    # X = lle.fit_transform(X)
    # Xt = lle.transform(Xt)

    # dim = 3
    # pca = PCA(n_components=dim)
    # X = pca.fit_transform(X)
    # Xt = pca.transform(Xt)
    # print(X.shape)
    # print(Xt.shape)

    # dim = 4
    # um = UMAP(n_components=dim, verbose=1)
    # X = um.fit_transform(X)
    # Xt = um.transform(Xt)

    # print(train_data)
    # sns.pairplot(train_data)
    # plt.show()
    X = torch.FloatTensor(X.astype("float64"))
    y = torch.FloatTensor(y.astype("float64")).unsqueeze(1)
    Xt = torch.FloatTensor(Xt.astype("float64"))

    print("get_data() done")
    return X, y, Xt, dim


def make_submission_file(y_pred):
    submission = pd.read_csv("./data/sample_submission.csv")
    submission["y"] = y_pred.detach().cpu()

    submission.to_csv("real_submission.csv", index=False)

    print("make_submission_file() done")


def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    pre_file_path = "real_submission.csv"
    if os.path.isfile(pre_file_path):
        os.remove(pre_file_path)

    X, y, Xt, dim = get_data()

    # model = nn.Linear(11, 1)
    model = Model(dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)
    X, y, Xt, = X.to(device), y.to(device), Xt.to(device)
    print(next(model.parameters()).device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    print("initial setting done")

    print(X.shape)
    print(Xt.shape)

    for epoch in range(1, 100001):
        output = model(X)
        cost = criterion(output, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch : {epoch}, Model : {list(model.parameters())}, Cost : {cost}")

        if cost < 100:
            break

    print(list(model.parameters()))
    y_pred = model(Xt)
    make_submission_file(y_pred)


if __name__ == "__main__":
    main()