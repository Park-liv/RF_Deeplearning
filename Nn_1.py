import numpy as np
import tensorflow as tf
import math
from batch_normalization import batch_norm

tf.set_random_seed(777)

num_test = 10000

data = np.load("processed_data.npy")
train_X_data = data[:-num_test, 0:-1]
train_Y_data = data[:-num_test, [-1]]
test_X_data = data[-num_test:, 0:-1]
test_Y_data = data[-num_test:, [-1]]
print(train_X_data.shape, train_Y_data.shape, test_X_data.shape, test_Y_data.shape)

nb_transmitter = 9
batch_size = 1000
num_epochs = 10
num_iterations = int(math.ceil(train_X_data.shape[0] / batch_size))

X = tf.placeholder(tf.float32, [None, 1999])
Y = tf.placeholder(tf.int32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

Y_one_hot = tf.one_hot(Y, nb_transmitter)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_transmitter])


W1 = tf.get_variable("W1", shape=[1999, 999],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([999]))
Bn1 = batch_norm(tf.matmul(X, W1) + b1)
L1 = tf.nn.relu(Bn1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[999, 499],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([499]))
Bn2 = batch_norm(tf.matmul(L1, W2) + b2)
L2 = tf.nn.relu(Bn2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[499, nb_transmitter],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([nb_transmitter]))
hypothesis = tf.matmul(L2, W3) + b3

# logits = tf.matmul(L2, W3) + b3
# hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

prediction = tf.argmax(hypothesis, axis=1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        avg_cost = 0
        avg_accuracy = 0

        for iteration in range(num_iterations):
            batch_xs = train_X_data[batch_size * iteration: batch_size * (iteration + 1), :]
            batch_ys = train_Y_data[batch_size * iteration: batch_size * (iteration + 1), :]
            _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += cost_val / num_iterations
            avg_accuracy += acc_val / num_iterations

        print(f"Epoch: {(epoch + 1):04d}, Cost: {avg_cost:.9f}, Accuracy: {avg_accuracy:.2%}")

    acc = sess.run(accuracy, feed_dict={X: test_X_data, Y: test_Y_data, keep_prob: 1})
    print(f"Accuracy: {(acc * 100):2.2f}%")

    show_X_data = test_X_data[-100:, :]
    show_Y_data = test_Y_data[-100:, :]

    pred = sess.run(prediction, feed_dict={X: show_X_data, keep_prob: 1})
    for p, y in zip(pred, show_Y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))