from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import pandas as pd
import pickle
import pymongo
import json
import os

class LoadData:
    def __init__(self):
        xs = 'pickles/x_scaler.pkl'
        ys = 'pickles/y_scaler.pkl'
        with open(xs, 'rb') as pickle_file:
            self.X_scaler = pickle.load(pickle_file)
        with open(ys, 'rb') as pickle_file:
            self.Y_scaler = pickle.load(pickle_file)

        mng_client = pymongo.MongoClient(
            "mongodb+srv://admin:admin@cluster0.vwurs.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
        db = mng_client.diabetes_data
        self.cursor = db.Diabetes_US.find({})

    # def __iter__(self):
    #     return self

    def next(self, batchsize=None):
        res = []
        counter = 0
        for r in self.cursor:
            res.append(r)
            counter += 1
            if batchsize and counter == batchsize:
                break
        df = pd.DataFrame(res).drop('_id', axis=1)

        X_training = df.drop('Outcome', axis=1).values
        Y_training = df[['Outcome']].values

        X_scaled = self.X_scaler.transform(X_training)
        Y_scaled = self.Y_scaler.transform(Y_training)

        return X_scaled, Y_scaled

def load_data_from_mongodb():
    mng_client = pymongo.MongoClient(
        "mongodb+srv://admin:admin@cluster0.vwurs.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    db = mng_client.diabetes_data
    res = db.Diabetes_US.find()
    print(res)
    return res


def import_content(filepath):
    # reading CSV and checking for null values
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)
    data = pd.read_csv(file_res)
    # print(data.info())
    if (data.isnull().sum().sum() != 0):
        print('Error: Found null values in the dataset.')
    else:
        print('There are no null values in the dataset.')

    # preparing database for upload
    mng_client = pymongo.MongoClient("mongodb+srv://admin:admin@cluster0.vwurs.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    mng_db = mng_client['diabetes_data'] # Replace mongo db name
    collection_name = 'Diabetes_US' # Replace mongo db collection name
    db_cm = mng_db[collection_name]
    data_json = json.loads(data.to_json(orient='records'))
    db_cm.remove()

    # inserting data to database
    db_cm.insert(data_json)
    print(db_cm.count_documents({}))

def load_data(scaled=True):
    datafile = "../diabetes.csv"

    df = shuffle(pd.read_csv(datafile ,dtype=float))

    training_data = df.sample(frac=0.9)
    X_training = training_data.drop('Outcome', axis=1).values
    Y_training = training_data[['Outcome']].values

    testing_data = df.drop(training_data.index)
    X_testing = testing_data.drop('Outcome', axis=1).values
    Y_testing = testing_data[['Outcome']].values

    if not scaled:
        return (X_training, Y_training), (X_testing, Y_testing)

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

    xs = 'pickles/x_scaler.pkl'
    ys = 'pickles/y_scaler.pkl'
    with open(xs, 'wb') as pickle_file:
        pickle.dump(X_scaler, pickle_file)
    with open(ys, 'wb') as pickle_file:
        pickle.dump(Y_scaler, pickle_file)

    return (X_scaled_training,Y_scaled_training), (X_scaled_testing, Y_scaled_testing)

if __name__ == '__main__':
    # import_content('../diabetes.csv')
    cursor = load_data_from_mongodb()
    for x in cursor:
        print(x)

