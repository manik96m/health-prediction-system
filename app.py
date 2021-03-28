
import os, sys
import pickle
from google.api_core.client_options import ClientOptions
import googleapiclient.discovery    # google-api-python-client
# from google.colab import auth

# 6,148,72,35,0,33.6,0.627,50
def predict_json(project, model, instances, version=None):
    endpoint = 'https://us-central1-ml.googleapis.com'
    client_options = ClientOptions(api_endpoint=endpoint)
    service = googleapiclient.discovery.build('ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    # print(instances)
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

    scaled_data = X_scaler.transform([instance]).tolist()

    # scaled_data = [[0.3529411764705882, 0.7437185929648241, 0.5901639344262295, 0.3535353535353536, 0.0, 0.5007451564828614,0.23441502988898377, 0.4833333333333334]]

    # print(instance)
    # print(scaled_data)

    project = 'my-project-first-296702'
    model = 'health_priddiction_system'
    version = 'v1'
    res = predict_json(project, model, scaled_data)
    print(res)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage example\n{}".format('python app.py "0,137,40,35,168,43.1,2.288,33"'))
    else:
        instance_str = sys.argv[1]
        instance = [float(s) for s in instance_str.split(",")]
        main(instance)