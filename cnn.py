import tensorflow as tf
import utils
import numpy as np

class CNN:
    def __init__(self):
        self.graph = tf.Graph()
        self._create_model()

    def _create_model(self):
        with self.graph.as_default():
            tf.reset_default_graph()
            with tf.name_scope('input'):
                self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1), name='input')
                self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout')
                self.label_placeholder = tf.placeholder(dtype=tf.int64, shape=(None, ), name='label')
            
            logits = None
            with tf.name_scope('model'):
                conv1 = tf.layers.conv2d(
                    inputs=self.input_placeholder,
                    filters=100,
                    kernel_size=5,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                    name="conv1")

                pool1 = tf.layers.max_pooling2d( 
                    inputs=conv1,
                    pool_size=2,
                    strides=2,
                    padding='valid',
                    name='pool1')

                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=100,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv2')

                pool2 = tf.layers.max_pooling2d( 
                    inputs=conv2,
                    pool_size=2,
                    strides=2,
                    padding='valid',
                    name='pool1')

                fc1_flat_input = tf.contrib.layers.flatten(pool2) 

                fc1 = tf.layers.dense(
                    inputs=fc1_flat_input,
                    units=1024,
                    activation=tf.nn.relu,
                    name='fc1')

                fc1 = tf.layers.dropout(fc1, 
                    rate=dropout,
                    training=self.keep_prob)

                fc2 = tf.layers.dense(
                    inputs=fc1,
                    units=128,
                    activation=tf.nn.relu,
                    name='fc2')

                logits = tf.layers.dense(
                    inputs=fc2,
                    units=10,
                    name='fc3')

            with tf.name_scope('evaluation'):
                self.probabilities = tf.nn.softmax(logits=logits, name='probabilities')
                self.predictions = tf.argmax(self.probabilities, axis=1, name='predicted_labels')
                self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.label_placeholder), tf.float32))

            return logits

    def _create_architecture(self, logits, learning_rate):
        with self.graph.as_default():
            with tf.name_scope('train'):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.label_placeholder,
                    logits=logits,
                    name='xentropy')
                self.loss = tf.reduce_mean(xentropy, name='loss')
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = optimizer.minimize(self.loss, name='train_op', global_step=tf.train.get_global_step())

    def train(self, train_set, validation_set, epoch, batch_size, learning_rate, dropout):
        with self.graph.as_default():
            logits = self._create_model()
            self._create_architecture(logits, learning_rate)

            with tf.Session() as sess:
                self.sess = sess
                self.sess.run(tf.global_variables_initializer())

                for epoch_no in range(epoch):
                    train_batch_data, train_batch_labels = utils._get_next_batch(train_set, batch_size)
                    val_batch_data, val_batch_labels = utils._get_next_batch(validation_set, batch_size)

                    _, train_loss, train_accuracy = self.sess.run([self.train_op, self.loss, self.accuracy], 
                        feed_dict={self.input_placeholder: train_batch_data, self.keep_prob: dropout, self.label_placeholder: train_batch_labels})
                    val_loss, val_accuracy = self.sess.run([self.loss, self.accuracy],
                        feed_dict={self.input_placeholder: val_batch_data, self.keep_prob: 1.0, self.label_placeholder: val_batch_labels})

                    print('\nEpoch: {}'.format(epoch_no + 1))
                    print('Train accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy / batch_size,
                                                                        train_loss / batch_size))
                    print('Val accuracy: {:.4f}, loss: {:.4f}\n'.format(val_accuracy / batch_size, 
                                                                        val_loss / batch_size))

    def eval(self, images):
        input_images = np.reshape(images, (images.shape[0], 28, 28, 1))

        with self.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, self.path)
                return sess.run([self.probabilities, self.predictions],
                        feed_dict={self.input_placeholder: input_images, self.is_training: False})

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        return
