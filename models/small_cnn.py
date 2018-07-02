import tensorflow as tf


initializer = tf.contrib.layers.variance_scaling_initializer
regularizer = tf.contrib.layers.l2_regularizer


def small_cnn(inputs, training, reuse=False, weight_decay=0):
    with tf.variable_scope("small_cnn", reuse=reuse):
        net = tf.layers.conv2d(
            inputs, filters=32, kernel_size=(3, 3),
            activation=tf.nn.relu,
            kernel_initializer=initializer(),
            kernel_regularizer=regularizer(scale=weight_decay))
        net = tf.layers.batch_normalization(net, axis=3, training=training,
                                            name="bn", fused=True)
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2))
        net = tf.layers.conv2d(
                net, filters=32, kernel_size=(3, 3),
                activation=tf.nn.relu,
                kernel_initializer=initializer(),
                kernel_regularizer=regularizer(scale=weight_decay))
        net = tf.layers.batch_normalization(net, axis=3, training=training,
                                            name="bn2", fused=True)
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2))
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, units=64, activation=tf.nn.relu)
        net = tf.layers.dropout(net, rate=0.5, training=training)
        return net
