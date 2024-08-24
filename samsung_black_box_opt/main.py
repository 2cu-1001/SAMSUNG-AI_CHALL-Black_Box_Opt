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
from copy import deepcopy


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(dim, 3)
        self.dropout1 = nn.Dropout(0.3)
        # self.layer2 = nn.Linear(16, 8)
        # self.dropout2 = nn.Dropout(0.3)
        # self.layer3 = nn.Linear(16, 8)
        # self.dropout3 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        # x = self.relu(self.layer2(x))
        # x = self.dropout2(x)
        # x = self.relu(self.layer3(x))
        # x = self.dropout3(x)
        return self.output_layer(x)


def get_data():
    # ------------------------------ load data ------------------------------#
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    # train_data = train_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)
    # test_data = test_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)
    # ------------------------------ load data ------------------------------#

    # ------------------------------ remove outlier ------------------------------#
    # sns.pairplot(train_data)
    # plt.show()
    # print(len(train_data))
    # plt.scatter([i for i in range(len(train_data))], train_data['y'], s=2)
    outlier = train_data[(abs((train_data['y'] - train_data['y'].mean()) / train_data['y'].std())) > 1.645].index
    train_data = train_data.drop(outlier)
    # plt.scatter([i for i in range(len(train_data))], train_data['y'], s=2, alpha=0.8)
    # plt.show()

    # sns.pairplot(train_data)
    # plt.show()
    X = train_data.values[:, 1:-1]
    y = train_data.values[:, -1]
    Xt = test_data.values[:, 1:]
    # ------------------------------ remove outlier ------------------------------#

    #------------------------------ dimension reduction ------------------------------#
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # Xt = scaler.transform(Xt)

    dim = 11

    # dim = 4
    # lle = LocallyLinearEmbedding(n_components=dim)
    # X = lle.fit_transform(X)
    # Xt = lle.transform(Xt)

    # pca = PCA(n_components=0.99)
    # X = pca.fit_transform(X)
    # Xt = pca.transform(Xt)
    # dim = pca.n_components_

    # dim = 8
    # um = UMAP(n_components=dim, verbose=1)
    # X = um.fit_transform(X)
    # Xt = um.transform(Xt)

    print(X.shape)
    print(Xt.shape)

    if dim == 2:
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(Xt[:, 0], Xt[:, 1])
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2])
        ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2])
        plt.show()
    # ------------------------------ dimension reduction ------------------------------#

    X = torch.FloatTensor(X.astype("float64"))
    y = torch.FloatTensor(y.astype("float64")).unsqueeze(1)
    Xt = torch.FloatTensor(Xt.astype("float64"))

    print("get_data() done")
    return X, y, Xt, dim


def make_submission_file(y_pred):
    submission = pd.read_csv("./data/sample_submission.csv")
    # submission["y"] = y_pred.detach().cpu()
    submission["y"] = y_pred

    submission.to_csv("real_submission.csv", index=False)

    print("make_submission_file() done")


def main():
    # ------------------------------ preprocessing ------------------------------#
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    pre_file_path = "real_submission.csv"
    if os.path.isfile(pre_file_path):
        os.remove(pre_file_path)

    X, y, Xt, dim = get_data()

    # model = nn.Linear(11, 1)
    # model = nn.Linear(dim, 1)
    model = Model(dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # ------------------------------ preprocessing ------------------------------#

    # ------------------------------ model init ------------------------------#
    model = model.to(device)
    nn.init.xavier_uniform_(model.layer1.weight)
    # nn.init.xavier_uniform_(model.layer2.weight)
    # nn.init.xavier_uniform_(model.layer3.weight)
    X, y, Xt, = X.to(device), y.to(device), Xt.to(device)
    print(next(model.parameters()).device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)
    print("initial setting done")

    print(X.shape)
    print(Xt.shape)
    # ------------------------------ model init ------------------------------#

    # ------------------------------ learning ------------------------------#
    for epoch in range(1, 1000001):
        model.train()
        output = model(X)
        cost = criterion(output, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch : {epoch}, Model : {list(model.parameters())}, Cost : {cost}")

        if cost < 5:
            break
    # ------------------------------ learning ------------------------------#

    # ------------------------------ predict ------------------------------#
    print(list(model.parameters()))
    y_pred = model(Xt)

    X = X.cpu().numpy()
    Xt = Xt.cpu().numpy()
    y = y.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    if dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y)
        ax.scatter(Xt[:, 0], Xt[:, 1], y_pred)
        plt.show()

    make_submission_file(y_pred)
    # ------------------------------ predict ------------------------------#

if __name__ == "__main__":
    main()