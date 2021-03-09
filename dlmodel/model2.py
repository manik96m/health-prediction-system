
import os
import tensorflow as tf
import datetime
from load_data import load_data

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_training, Y_training), (X_testing, Y_testing) = load_data(True)

model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(8,)),
  tf.keras.layers.Flatten(input_dim=8),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(10, activation='sigmoid'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])

# predictions = model(X_training[:1]).numpy()
# print(predictions)
# print(tf.nn.softmax(predictions).numpy())
#
# ‘binary_crossentropy‘ for binary classification.
# ‘sparse_categorical_crossentropy‘ for multi-class classification.
# ‘mse‘ (mean squared error) for regression.

# loss_fn = tf.keras.losses.binary_crossentropy(from_logits=True)
# loss_fn(Y_training[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


log_dir = "logs2/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_training, Y_training, batch_size=100, epochs=250, callbacks=[tensorboard_callback])

# model.evaluate(X_testing[:2],  Y_testing[:2], verbose=2)
loss, acc = model.evaluate(X_testing, Y_testing, verbose=0)
print('Test Accuracy: %.3f, loss: %.3f' % (acc, loss))
# Test Accuracy: 0.792, loss: 0.463
#
# print(Y_testing[:10])
# print(model.predict(X_testing[:10]))

