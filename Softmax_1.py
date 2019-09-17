from Data_processing import combined_data
import numpy as up
import tensorflow as tf
import random

data = combined_data
train_X_data = data[0:6449, 0:-1]
train_Y_data = data[0:6449, [-1]]
test_X_data = data[6449:, 0:-1]
test_Y_data = data[6449:, [-1]]
# train_Y_data = train_Y_data.astype('int32')
# test_Y_data = test_Y_data.astype('int32')

print(train_X_data.shape, train_Y_data.shape, test_X_data.shape, test_Y_data.shape)

nb_transmitter = 3

# tf.set_random_seed(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

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

    pred = sess.run(prediction, feed_dict={X: test_X_data})
    for p, y in zip(pred, test_Y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

'''
W1 = tf.Variable(tf.random_normal([1999, 1024]))
b1 = tf.Variable(tf.random_normal([1024]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([1024, 512]))
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([512, 128]))
b3 = tf.Variable(tf.random_normal([128]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.Variable(tf.random_normal([128, 3]))
b4 = tf.Variable(tf.random_normal([3]))
hypothesis = tf.matmul(L3, W4) + b4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
'''