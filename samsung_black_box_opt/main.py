from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import pandas as pd
import os


def get_data():
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    X = train_data.values[:, 1:-1]
    y = train_data.values[:, -1]
    Xt = test_data.values[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    #return X_train, X_test, y_train, y_test, Xt

    print("get_data() done")
    return X, y, Xt

def preprocessing(X, Xt):
    # stdsc = StandardScaler()
    # X_train_std = stdsc.fit_transform(X_train)
    # X_test_std = stdsc.transform(X_test)
    # Xt_std = stdsc.transform(Xt)

    # pca = PCA(n_components=8)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)
    # Xt_pca = pca.transform(Xt)

    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X)
    Xt_pca = pca.transform(Xt)

    # return X_train_std, X_test_std, Xt_std
    # return X_train_pca, X_test_pca, Xt_pca

    print("preprocessing() done")
    return X_pca, Xt_pca


def learning(X, y, Xt):
    # poly = PolynomialFeatures(degree=8, include_bias=False)
    # X_poly = poly.fit_transform(X_train)
    # X_test_poly = poly.fit_transform(X_test)
    # Xt_poly = poly.fit_transform(Xt)
    #
    # lr = Ridge(alpha = 0.01)
    # lr.fit(X_poly, y_train)
    # print('Training accuracy:', lr.score(X_poly, y_train))
    # print('Test accuracy:', lr.score(X_test_poly, y_test))
    # y_pred = lr.predict(Xt_poly)

    # lr = LinearRegression()
    # lr.fit(X_train, y_train)
    # print('Training accuracy:', lr.score(X_train, y_train))
    # print('Test accuracy:', lr.score(X_test, y_test))
    # y_pred = lr.predict(Xt)

    lr = RandomForestRegressor()
    lr.fit(X, y)
    y_pred = lr.predict(Xt)
    print('Training accuracy:', lr.score(X, y))

    print("learning() done")
    return y_pred


def make_submission_file(y_pred):
    submission = pd.read_csv("./data/sample_submission.csv")
    submission["y"] = y_pred

    submission.to_csv("real_submission.csv", index=False)

    print("make_submission_file() done")

def main():
    pre_file_path = "real_submission.csv"
    if os.path.isfile(pre_file_path):
        os.remove(pre_file_path)

    # X_train, X_test, y_train, y_test, Xt = get_data()
    X, y, Xt = get_data()

    #X_train X_test, Xt = preprocessing(X_train, X_test, Xt)
    X, Xt = preprocessing(X, Xt)

    #y_pred = learning(X_train, X_test, y_train, y_test, Xt)
    y_pred = learning(X, y, Xt)

    make_submission_file(y_pred)


if __name__ == "__main__":
    main()
