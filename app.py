
import os, sys
import pickle

# 6,148,72,35,0,33.6,0.627,50
def main(healthdata):
    xs = 'pickles/x_scaler.pkl'
    ys = 'pickles/y_scaler.pkl'
    with open(xs, 'rb') as pickle_file:
        X_scaler = pickle.load(pickle_file)
    with open(ys, 'rb') as pickle_file:
        Y_scaler = pickle.load(pickle_file)

    scaled_data = X_scaler.transform([healthdata])


if __name__ == "__main__":
    main(sys.argv[1:])