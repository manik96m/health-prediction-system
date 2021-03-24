from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import pandas as pd
import pickle
import pymongo
import json
import os

def load_data_from_mongodb():
    mng_client = pymongo.MongoClient(
        "mongodb+srv://admin:admin@cluster0.vwurs.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    db = mng_client.diabetes_data
    # collection = db.Diabetes_US
    print(db.Diabetes_US.find_one({})) # {'Pregnancies':1}


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

    # xs = 'pickles/x_scaler.pkl'
    # ys = 'pickles/y_scaler.pkl'
    # with open(xs, 'wb') as pickle_file:
    #     pickle.dump(X_scaler, pickle_file)
    # with open(ys, 'wb') as pickle_file:
    #     pickle.dump(Y_scaler, pickle_file)

    return (X_scaled_training,Y_scaled_training), (X_scaled_testing, Y_scaled_testing)


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

if __name__ == '__main__':
    import_content('../diabetes.csv')


