import os
import tensorflow as tf
from load_data import load_data


# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define computational grph
X = [1,1,2]
Y = [4,2,3]

addition = tf.add(X, Y)

print(addition)

(X_scaled_training,Y_scaled_training), (X_scaled_testing, Y_scaled_testing) = load_data()

learning_rate = 0.001
training_epochs = 10

number_of_inputs = 9
number_of_outputs = 1

layer_1_nodes = 10
layer_2_nodes = 50
layer_3_nodes = 30

with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weight1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weight2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weight3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

with tf.variable_scope('output'):
    weights = tf.get_variable("weight4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

# Define the cost function that will measure prediction accuracy during training.

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.AdamOptimizer(learning_rate).minimize(cost)

# Launch a session to run TensorFlow operations
with tf.Session() as session:
    # Run the global variable initializer to initialize all variables and layers of ther neural network
    session.run(tf.global_variables_initializer())

    # Run the optimizer over and over again to train the network.
    for epoch in range(training_epochs):
        # Feed the training data
        session.run(optimizer, feed_dict={X: X_scaled_training, Y:Y_scaled_training})

        # log the progress for every 5 epochs.
        if epoch % 5 == 0:
            training_cost = tf.run(cost, feed_dict={X: X_scaled_training, Y:Y_scaled_training})
            testing_cost = tf.run(cost, feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})
            print("epoch #{} - training_cost {}, testing_cost {}".format(epoch, training_cost, testing_cost))

