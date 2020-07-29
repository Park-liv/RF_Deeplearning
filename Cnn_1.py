import numpy as np
import tensorflow as tf
import math
from batch_normalization import batch_norm

tf.set_random_seed(777)

num_test = 10000

data = np.load("1M_data.npy")
data2 = np.load("2M_data.npy")
data3 = np.concatenate((data, data2), axis=0)
np.random.shuffle(data3)
np.random.shuffle(data3)
np.random.shuffle(data3)
train_X_data = data3[:-num_test, 0:-1]
train_Y_data = data3[:-num_test, [-1]]
test_X_data = data3[-num_test:, 0:-1]
test_Y_data = data3[-num_test:, [-1]]
print(train_X_data.shape, train_Y_data.shape, test_X_data.shape, test_Y_data.shape)

nb_transmitter = 9
batch_size = 500
num_epochs = 10
num_iterations = int(math.ceil(train_X_data.shape[0] / batch_size))

X = tf.placeholder(tf.float32, [None, 1799])
Y = tf.placeholder(tf.int32, [None, 1])

X_con = tf.reshape(X, [-1, 1799, 1])
# print("X_con's shape is :", X_con.shpae)

keep_prob = tf.placeholder(tf.float32)

Y_one_hot = tf.one_hot(Y, nb_transmitter)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_transmitter])

W1 = tf.Variable(tf.random_normal([20, 1, 8], stddev=0.01))
L1 = tf.nn.conv1d(X_con, W1, stride=[1, 2, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.pool(L1, pooling_type='MAX',window_shape=[4], strides=[4], padding='SAME')
print(L1.shape)
'''
    1999 (?, 250, 8)
    1799 (?, 225, 8)
'''
# L1_flat = tf.reshape(L1, [-1, 225 * 8])

W2 = tf.Variable(tf.random_normal([20, 8, 16], stddev=0.01))
L2 = tf.nn.conv1d(L1, W2, stride=[1, 2, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.pool(L2, pooling_type='MAX',window_shape=[2], strides=[2], padding='SAME')
print(L2.shape)
'''
    (?, 57, 16)
'''
W3 = tf.Variable(tf.random_normal([20, 16, 32], stddev=0.01))
L3 = tf.nn.conv1d(L2, W3, stride=[1, 2, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.pool(L3, pooling_type='MAX',window_shape=[2], strides=[2], padding='SAME')
print(L3.shape)
'''
    (?, 15, 32)
'''
L3_flat = tf.reshape(L3, [-1, 15 * 32])

W4 = tf.get_variable("W4", shape=[15 * 32, 500],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([500]))
Bn4 = batch_norm(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.relu(Bn4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[500, nb_transmitter],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_transmitter]))
hypothesis = tf.matmul(L4, W5) + b5

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
            _, cost_val, acc_val = sess.run([train, cost, accuracy],
                                            feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
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