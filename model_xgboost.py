import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def build_model(filepath):
    # splitting dataset (X=independent variables, Y=dependent variable)
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)
    data = pd.read_csv(file_res)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    # imputing zero values by using mean
    values = SimpleImputer(missing_values=0, strategy='mean')
    X_train = values.fit_transform(X_train)
    X_test = values.fit_transform(X_test)

    # training the model
    xgb_model = XGBClassifier(gamma=0, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    Xnew = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
    pred = xgb_model.predict(np.array(Xnew).reshape((1, -1)))
    print(pred)

    # print(X_test)

    print("Accuracy Score for the model =", format(
        metrics.accuracy_score(y_test, xgb_pred)))


if __name__ == "__main__":
    filepath = 'diabetes.csv'  # pass csv file path
    build_model(filepath)
