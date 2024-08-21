import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from copy import deepcopy


def get_data():
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    # train_data = train_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)
    # test_data = test_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)

    X = train_data.values[:, 1:-1]
    y = train_data.values[:, -1]
    Xt = test_data.values[:, 1:]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xt = scaler.transform(Xt)

    pca = PCA(n_components=0.7)
    X = pca.fit_transform(X)
    dim = pca.n_components_

    Xt = pca.transform(Xt)
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
    model = nn.Linear(dim, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)
    X, y, Xt, = X.to(device), y.to(device), Xt.to(device)
    print(next(model.parameters()).device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    print("initial setting done")

    print(X.shape)
    print(Xt.shape)

    for epoch in range(1, 250001):
        output = model(X)
        cost = criterion(output, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch : {epoch}, Model : {list(model.parameters())}, Cost : {cost}")

        if cost < 1:
            break

    print(list(model.parameters()))
    y_pred = model(Xt)

    if dim == 2:
        X = X.cpu().numpy()
        Xt = Xt.cpu().numpy()
        y = y.cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y)
        ax.scatter(Xt[:, 0], Xt[:, 1], y_pred)
        plt.show()

    make_submission_file(y_pred)


if __name__ == "__main__":
    main()