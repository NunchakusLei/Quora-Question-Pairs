import tensorflow as tf
import pandas as pd
import numpy as np
from read_datas import *

def main(_):
    # Import data
    df = pd.read_csv("data/train.csv")

    # Create the model
    x = tf.placeholder(tf.float32, [None, 54])
    W = tf.Variable(tf.zeros([54, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Feature extraction
    feature_M, labels_M = preproceesing(df)
    # print feature_M[0:2], labels_M[0:2]

    # Train
    for i in range(100):
        batch_xs, batch_ys = feature_M[i*100:(i+1)*100], labels_M[i*100:(i+1)*100]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acry_rate_t = 0.0
    for j in range((len(labels_M)-10000)//100):
        i += 1
        acry_rate = sess.run(accuracy, feed_dict={x: feature_M[i*100:(i+1)*100],
                                            y_: labels_M[i*100:(i+1)*100]})
        acry_rate_t += acry_rate
        # print(acry_rate)
    print("-- Accuracy in total:",acry_rate_t/j)


if __name__ == '__main__':
    tf.app.run(main=main)
