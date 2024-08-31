from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import pandas as pd
import os


def get_data():
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    train_data = train_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)
    test_data = test_data.drop(['x_0', 'x_1', 'x_2', 'x_3'], axis=1)

    X = train_data.values[:30000, 1:-1]
    y = train_data.values[:30000, -1]
    Xt = test_data.values[:, 1:]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("get_data() done")
    return X, y, Xt


def preprocessing(X, y, Xt):
    stdsc = StandardScaler()
    X = stdsc.fit_transform(X)
    Xt = stdsc.transform(Xt)

    svr = SVR(degree=4)
    svr.fit(X, y)
    y_pred = svr.predict(Xt)

    return y_pred


    # pca = PCA(n_components=8)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)
    # Xt_pca = pca.transform(Xt)

    print("preprocessing() done")
    # return X_train_std, X_test_std, Xt_std
    return X_train_pca, X_test_pca, Xt_pca


def learning(X_train, X_test, y_train, y_test, Xt):
    poly = PolynomialFeatures(degree=8, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    Xt_poly = poly.fit_transform(Xt)

    lr = Ridge(alpha=0.01)
    lr.fit(X_poly, y_train)
    print('Training accuracy:', lr.score(X_poly, y_train))
    print('Test accuracy:', lr.score(X_test_poly, y_test))
    y_pred = lr.predict(Xt_poly)

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

    X, y, Xt = get_data()

    y_pred = preprocessing(X, y, Xt)

    # y_pred = learning(X_train_std, X_test_std, y_train, y_test, Xt_std)

    make_submission_file(y_pred)


if __name__ == "__main__":
    main()