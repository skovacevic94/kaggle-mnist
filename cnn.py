import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self):
        self.graph = tf.Graph()

    def _create_model(self, learning_rate):
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1), name='input')
                self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='label')
            
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
                    kernel_size=4,
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

                fc1_flat_input = tf.reshape(pool2, shape=(-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3])) 

                fc1 = tf.layers.dense(
                    inputs=fc1_flat_input,
                    units=300,
                    activation=tf.nn.relu,
                    name='fc1')

                fc2 = tf.layers.dense(
                    inputs=fc1,
                    units=100,
                    activation=tf.nn.relu,
                    name='fc2')

                logits = tf.layers.dense(
                    inputs=fc2,
                    units=10,
                    activation=tf.nn.relu,
                    name='fc3')

            with tf.name_scope('train'):
                xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.label_placeholder,
                    logits=logits,
                    name='xentropy')
                self.loss = tf.reduce_mean(xentropy, name='loss')
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = optimizer.minimize(self.loss, name='train_op')

            with tf.name_scope('evaluation'):
                self.probabilities = tf.nn.softmax(logits=logits, name='probabilities')
                self.predictions = tf.argmax(self.probabilities, axis=1, name='predicted_labels')
                self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predictions, tf.argmax(self.label_placeholder, axis=1)), tf.float32))
    
    @staticmethod
    def _get_next_batch(dataframe, batch_size):
        batch_dataframe = dataframe.sample(batch_size)

        data = np.reshape(batch_dataframe.drop(['label'], axis=1).values, (batch_size, 28, 28, 1))
        labels = np.reshape(batch_dataframe[['label']].values, (batch_size))
        
        labels_onehot = np.eye(10)[labels]
        
        return data, labels_onehot

    def train(self, train_set, validation_set, epoch, batch_size, learning_rate):
        with self.graph.as_default():
            self._create_model(learning_rate)
            
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for epoch_no in range(epoch):
                    train_batch_data, train_batch_labels = CNN._get_next_batch(train_set, batch_size)
                    val_batch_data, val_batch_labels = CNN._get_next_batch(validation_set, batch_size)

                    _, train_loss, train_accuracy = sess.run([self.train_op, self.loss, self.accuracy], 
                        feed_dict={self.input_placeholder: train_batch_data, self.label_placeholder: train_batch_labels})
                    val_loss, val_accuracy = sess.run([self.loss, self.accuracy],
                        feed_dict={self.input_placeholder: val_batch_data, self.label_placeholder: val_batch_labels})

                    print('\nEpoch: {}'.format(epoch_no + 1))
                    print('Train accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy / batch_size,
                                                                        train_loss / batch_size))
                    print('Val accuracy: {:.4f}, loss: {:.4f}\n'.format(val_accuracy / batch_size, 
                                                                        val_loss / batch_size))
                saver.save(sess, '/tmp/model.ckpt')

    def eval(self, image):
        input_image = np.reshape(image, (1, 28, 28, 1))

        with self.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, '/tmp/model.ckpt')
                return sess.run([self.probabilities, self.predictions],
                        feed_dict={self.input_placeholder: input_image})
