
import os
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
import datetime
from load_data import load_data
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOG_DIR = f"tuner_logs/{datetime.datetime.now().timestamp()}"

(X_training, Y_training), (X_testing, Y_testing) = load_data(True)

def build_model(hp):
    model = keras.models.Sequential()
    model.add(Dense(hp.Int("input_units", min_value=16, max_value=160, step=16),
                     input_shape=X_training.shape[1:]))
    for i in range(hp.Int("num_layers", min_value=1, max_value=6, step=2)):
        model.add(Dense(hp.Int(f"units_{i}", min_value=12, max_value=24, step=4), activation=keras.activations.relu))
    model.add(Dense(1, activation=keras.activations.sigmoid))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
        loss=keras.losses.binary_crossentropy,
        metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model, objective="val_accuracy", max_trials=2, executions_per_trial=2, directory=LOG_DIR)
tuner.search(X_training, Y_training, epochs=20, validation_data=(X_testing, Y_testing))

models = tuner.get_best_models(num_models=2)
# print(tuner.results_summary())
print(models[0].predict(X_testing[:10]))
models[0].save('mymodel_'+str(datetime.datetime.now()).replace("-", "").replace(" ", "").replace(":","").replace(".",""))
print(Y_testing[:10])
