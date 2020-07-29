<<<<<<< HEAD
import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

data = np.load("processed_data.npy")
train_X_data = data[:-10000, 0:-1]
train_Y_data = data[:-10000, [-1]]
test_X_data = data[-10000:, 0:-1]
test_Y_data = data[-10000:, [-1]]

print(train_X_data.shape, train_Y_data.shape, test_X_data.shape, test_Y_data.shape)
print(train_Y_data)

nb_transmitter = 9

X = tf.placeholder(tf.float32, [None, 1999])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_transmitter)
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_transmitter])
print("reshape one_hot:", Y_one_hot)

W = tf.Variable(tf.random_normal([1999, nb_transmitter]), name='weight')
b = tf.Variable(tf.random_normal([nb_transmitter]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: train_X_data, Y: train_Y_data})

        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    show_X_data = test_X_data[-100:, :]
    show_Y_data = test_Y_data[-100:, :]

    pred = sess.run(prediction, feed_dict={X: show_X_data})
    for p, y in zip(pred, show_Y_data.flatten()):
=======
import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

data = np.load("processed_data.npy")
train_X_data = data[:-10000, 0:-1]
train_Y_data = data[:-10000, [-1]]
test_X_data = data[-10000:, 0:-1]
test_Y_data = data[-10000:, [-1]]

print(train_X_data.shape, train_Y_data.shape, test_X_data.shape, test_Y_data.shape)
print(train_Y_data)

nb_transmitter = 9

X = tf.placeholder(tf.float32, [None, 1999])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_transmitter)
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_transmitter])
print("reshape one_hot:", Y_one_hot)

W = tf.Variable(tf.random_normal([1999, nb_transmitter]), name='weight')
b = tf.Variable(tf.random_normal([nb_transmitter]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: train_X_data, Y: train_Y_data})

        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    show_X_data = test_X_data[-100:, :]
    show_Y_data = test_Y_data[-100:, :]

    pred = sess.run(prediction, feed_dict={X: show_X_data})
    for p, y in zip(pred, show_Y_data.flatten()):
>>>>>>> 415b628f98856795908f91cbd80e1edf79075e5a
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))