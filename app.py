
import os, sys
import pickle
import googleapiclient.discovery
# from google.colab import auth

# def explicit():
#     from google.cloud import storage
#
#     # Explicitly use service account credentials by specifying the private key
#     # file.
#     storage_client = storage.Client.from_service_account_json(
#         'service_account.json')
#
#     # Make an authenticated API request
#     buckets = list(storage_client.list_buckets())
#     print(buckets)

# 6,148,72,35,0,33.6,0.627,50
def predict_json(project, model, instances, version=None):
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    print(instances)
    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

def main(instance):
    xs = 'dlmodel/pickles/x_scaler.pkl'
    ys = 'dlmodel/pickles/y_scaler.pkl'
    with open(xs, 'rb') as pickle_file:
        X_scaler = pickle.load(pickle_file)
    with open(ys, 'rb') as pickle_file:
        Y_scaler = pickle.load(pickle_file)

    # auth.authenticate_user()

    scaled_data = X_scaler.transform([instance]).tolist()
    # print(instance)
    # print(scaled_data)

    project = 'my-project-first-296702'
    # project = 'My Project first'
    model = 'health_priddiction_system'
    version = 'v1'
    predict_json(project, model, scaled_data, version)


if __name__ == "__main__":
    instance_str = sys.argv[1]
    instance = [float(s) for s in instance_str.split(",")]
    main(instance)