import sys
import pandas as pd
import pymongo
import json
import os


def import_content(filepath):
    mng_client = pymongo.MongoClient("mongodb+srv://admin:admin@cluster0.vwurs.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    mng_db = mng_client['diabetes_data'] # Replace mongo db name
    collection_name = 'diabetes_collection' # Replace mongo db collection name
    db_cm = mng_db[collection_name]
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)

    data = pd.read_csv(file_res)
    data_json = json.loads(data.to_json(orient='records'))
    db_cm.remove()
    db_cm.insert(data_json)
    print(db_cm.count_documents({}))

if __name__ == "__main__":
    filepath = 'diabetes_data_upload.csv'  # pass csv file path
    import_content(filepath)