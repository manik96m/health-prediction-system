from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import pandas as pd
import pickle

def load_data(scaled=False):
    datafile = "../diabetes.csv"

    df = shuffle(pd.read_csv(datafile ,dtype=float))

    training_data = df.sample(frac=0.9)
    X_training = training_data.drop('Outcome', axis=1).values
    Y_training = training_data[['Outcome']].values

    testing_data = df.drop(training_data.index)
    X_testing = testing_data.drop('Outcome', axis=1).values
    Y_testing = testing_data[['Outcome']].values

    if not scaled:
        return (X_training, Y_training), (X_testing, Y_testing), None

    print(len(df.index))
    print(len(training_data.index))
    print(len(testing_data.index))
    print(df.columns)

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled_training = X_scaler.fit_transform(X_training)
    Y_scaled_training = Y_scaler.fit_transform(Y_training)

    X_scaled_testing = X_scaler.transform(X_testing)
    Y_scaled_testing = Y_scaler.transform(Y_testing)
    # print(X_scaled_training)

    xs = 'tmp/x_scaler.pkl'
    ys = 'tmp/y_scaler.pkl'
    pickle.dump(X_scaler, xs)
    pickle.dump(Y_scaler, ys)

    return (X_scaled_training,Y_scaled_training), (X_scaled_testing, Y_scaled_testing)


