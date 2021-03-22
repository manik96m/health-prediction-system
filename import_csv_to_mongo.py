
import pandas as pd
import pymongo
import json
import os


def import_content(filepath):
    # reading CSV and checking for null values
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)
    data = pd.read_csv(file_res)
    print(data.info())
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


if __name__ == "__main__":
    filepath = 'diabetes.csv'  # pass csv file path
    import_content(filepath)
