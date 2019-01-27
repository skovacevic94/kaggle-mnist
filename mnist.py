import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

input_layer = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
labels = tf.placeholder(dtype=tf.int64, shape=(None))

conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=64,
    kernel_size=5,
    strides=1,
    padding="same",
    activation=tf.nn.relu,
    name="conv1")

pool1 = tf.layers.max_pooling2d( 
    inputs=conv1,
    pool_size=4,
    strides=2,
    padding='valid',
    name='pool1')

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=120,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu,
    name='conv2')

pool2 = tf.layers.max_pooling2d( 
    inputs=conv2,
    pool_size=4,
    strides=2,
    padding='valid',
    name='pool1')

conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=50,
    kernel_size=10,
    strides=1,
    padding='same',
    activation=tf.nn.relu,
    name='conv3')

fc1_flat_input = tf.reshape(conv3, shape=(-1, conv3.shape[1]*conv3.shape[2]*conv3.shape[3])) #shape=(None, 7*7*20) 

fc1 = tf.layers.dense( #shape=(None, 1024)
    inputs=fc1_flat_input,
    units=1024,
    activation=tf.nn.relu,
    name='fc1')

fc2 = tf.layers.dense( #shape=(None, 10)
    inputs=fc1,
    units=10,
    activation=tf.nn.relu,
    name='fc2')

prob = tf.nn.softmax( #shape=(None, 1024)
    logits=fc1,
    name='prob')

predicted_label = tf.argmax(prob, axis=1) #shape=(None, 1)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits( #shape=(None, 1)
    labels=labels,
    logits=fc1,
    name='xentropy')

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(0.0001)
training_op = optimizer.minimize(loss)

correct = tf.equal(predicted_label, labels)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

batch_size = 100

raw_dataframe = pd.read_csv("D:\\Data\\mnist-kaggle\\train.csv")
raw_train, raw_test = train_test_split(raw_dataframe, test_size=0.3)

label_test = np.reshape(raw_test[['label']].values, raw_test.shape[0])
image_test = np.reshape(raw_test.drop(['label'], axis=1).values, (raw_test.shape[0], 28, 28, 1))

with tf.Session() as sess:
    init.run()
    for epoch in range(300):
        sample_raw = raw_train.sample(batch_size)
        label_batch = np.reshape(sample_raw[['label']].values, (batch_size))
        image_batch = np.reshape(sample_raw.drop(['label'], axis=1).values, (batch_size, 28, 28, 1))

        sess.run([training_op], feed_dict={input_layer:image_batch, labels:label_batch})
        acc_train = accuracy.eval(feed_dict={input_layer:image_batch, labels:label_batch})
    acc_test = accuracy.eval(feed_dict={input_layer:image_test, labels:label_test})
    print('Epoch:', epoch+1, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)