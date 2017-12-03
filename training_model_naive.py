import tensorflow as tf
import pandas as pd
import numpy as np
from read_datas import *


def create_model_structure(input_shape):
    x = tf.placeholder(tf.float32, input_shape)

    # W = tf.Variable(tf.zeros([input_shape[1], 1]))
    # b = tf.Variable(tf.zeros([1]))

    W = tf.Variable(tf.truncated_normal([input_shape[1], 40], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[40]))

    h = tf.sigmoid(tf.matmul(x, W) + b)

    # W2 = tf.Variable(tf.zeros([40, 1]))
    # b2 = tf.Variable(tf.zeros([1]))
    W2 = tf.Variable(tf.truncated_normal([40, 1], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))

    y = tf.sigmoid(tf.matmul(h, W2) + b2)

    # y = tf.matmul(x, W) + b
    # y = tf.sigmoid(y)
    return x, y


def define_loss_fn(y_, y):
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    cross_entropy = tf.reduce_mean(
        tf.losses.log_loss(labels=y_, predictions=y))
    # cross_entropy = -tf.reduce_sum(
    #       y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    # cross_entropy = tf.reduce_mean(
    #       tf.losses.mean_squared_error(labels=y_, predictions=y))
    return cross_entropy


def main(_):
    # Import data
    df = pd.read_csv("data/train.csv")
    # Feature extraction
    feature_M, labels_M = preproceesing(df)
    input_dimension = feature_M.shape[1]
    # print feature_M[0:2], labels_M[0:2]

    # Create the model
    x, y = create_model_structure([None, input_dimension])
    # Create label variable
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Define loss and optimizer
    loss = define_loss_fn(y_, y)

    estimator = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    # Evaluation method
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # correct_prediction = tf.losses.absolute_difference(y_, y)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(
        tf.losses.log_loss(labels=y_, predictions=y))

    epoch = 100
    episode = 200
    Error = np.float('inf')
    # Train
    for i in range(episode):
        batch_xs, batch_ys = feature_M[i*epoch:(i+1)*epoch], labels_M[i*epoch:(i+1)*epoch]

        # train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})#, keep_prob: 1.0 })
        print("step %d, training loss %g"%(i, Error))

        _, Error = sess.run([estimator, loss], \
                            feed_dict={x: batch_xs, y_: batch_ys})

    # Test model
    acry_rate_t = 0.0
    for j in range((len(labels_M)-(episode*epoch))//epoch):
        i += 1
        acry_rate = sess.run(accuracy, feed_dict={x: feature_M[i*epoch:(i+1)*epoch],
                                            y_: labels_M[i*epoch:(i+1)*epoch]})
        if not np.isnan(acry_rate):
            acry_rate_t += acry_rate
        # print(acry_rate)
    print("-- Accuracy in total:",acry_rate_t/j)

    # Try to predict
    i = 0
    classification = sess.run(y, feed_dict={x: feature_M[i*10:(i+1)*10]})
    print 'NN predicted', classification
    print 'Ground truth', labels_M[i*10:(i+1)*10]


if __name__ == '__main__':
    tf.app.run(main=main)
