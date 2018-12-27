import tensorflow as tf
import numpy as np

"""
Main graph nodes are defined here.
"""

def variable_summaries(var):
    """
    Useful parameters to analyze a variable.
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def conv(x, training, D, K, F, S, scope, activation=True):
    """
    Wrapper for convolutional layer.
    """
    with tf.variable_scope(scope):
        strides = [1, S, S, 1]
        filter = tf.get_variable('weights', [F,F,D,K], initializer =
                 tf.truncated_normal_initializer(stddev=0.01))
        variable_summaries(filter)
        bias = tf.get_variable('bias', [K], initializer = tf.constant_initializer(0))
        # variable_summaries(bias)
        x = tf.nn.bias_add(tf.nn.conv2d(x, filter, strides, padding = 'SAME', name=scope), bias)
        with tf.variable_scope('batch_norm'):
            x = tf.layers.batch_normalization(x)
            tf.summary.histogram("batch_norm",x)
        if(activation):
            x = tf.nn.relu(x)
        tf.summary.histogram(scope, x)
        return x

def max_pool(x, F, S, scope):
    """
    Wrapper for 2D pooling. Also returns pooling indices.
    """
    with tf.variable_scope(scope):
        strides = [1,S,S,1]
        op,indices = tf.nn.max_pool_with_argmax(x, [1, F, F, 1], strides, padding = 'SAME')
        tf.summary.histogram(scope, op)
        return op, indices

def unpool(x, output_shape,D,K,F, S, scope):
    """
    Wrapper for transpose convolution.
    """
    with tf.variable_scope(scope):
        filter = tf.get_variable('weights', [F,F,D,K],
             initializer = tf.truncated_normal_initializer(stddev=0.01))
        variable_summaries(filter)
        strides = [1,S,S,1]
        op = tf.nn.conv2d_transpose(x, filter, output_shape,strides,padding = 'SAME')
        tf.summary.histogram(scope, op)
        return op

def loss(logits, labels):
    """
    Cross entropy loss with softmax.
    """
    one_hot_labels = tf.one_hot(labels, depth=11)
    with tf.variable_scope("cross_entropy"):
        ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels,logits=logits, name='cross_entropy')
        ce_mean = tf.reduce_mean(ce_loss, name='ce_mean')
        tf.summary.scalar('cross_entropy', ce_mean)
        return ce_mean

def accuracy(preds, labels):
    print(preds, labels)
    acc = tf.math.equal(preds, labels)
    acc = tf.reduce_sum(tf.cast(acc, tf.float32))
    acc = acc/(360*480)
    return acc

def predict(image, training):
    """
    Returns unnormalized predictions (no activation at final layer)
    """
    with tf.variable_scope("model"):
        img = tf.nn.lrn(image, depth_radius=5, bias=0.01, alpha=0.0001, beta=0.75, name='norm_img')

        conv1a = conv(img, training, D=3, K=64, F=7, S=1, scope = "conv1a")
        conv1b = conv(conv1a, training, D=64, K=64, F=7, S=1, scope = "conv1b")
        pool1, pool1_indices = max_pool(conv1b, F=2, S=2, scope = "pool1")

        conv2a = conv(pool1, training, D=64, K=64, F=7, S=1, scope = "conv2a")
        conv2b = conv(conv2a, training, D=64, K=64, F=7, S=1, scope = "conv2b")
        pool2, pool2_indices = max_pool(conv2b, F=2, S=2, scope = "pool2")

        conv3a = conv(pool2, training, D=64, K=64, F=7, S=1, scope = "conv3a")
        conv3b = conv(conv3a, training, D=64, K=64, F=7, S=1, scope = "conv3b")
        pool3, pool3_indices = max_pool(conv3b, F=2, S=2, scope = "pool3")

        conv4a = conv(pool3, training, D=64, K=64, F=7, S=1, scope = "conv4a")
        conv4b = conv(conv4a, training, D=64, K=64, F=7, S=1, scope = "conv4b")
        pool4, pool4_indices = max_pool(conv4b, F=2, S=2, scope = "pool4")

        unpool4 = unpool(pool4, conv4b.get_shape(),64,64,F=2, S=2, scope="unpool4")
        deconv4b = conv(unpool4, training, D = 64, K=64, F=7, S=1, scope = "deconv4b")
        deconv4a = conv(deconv4b, training, D = 64, K=64, F=7, S=1, scope = "deconv4a")

        unpool3 = unpool(deconv4a, conv3b.get_shape(),64,64,F=2, S=2, scope="unpool3")
        deconv3b = conv(unpool3, training, D=64, K=64, F=7, S=1, scope = "deconv3b")
        deconv3a = conv(deconv3b, training, D=64, K=64, F=7, S=1, scope = "deconv3a")

        unpool2 = unpool(deconv3a, conv2b.get_shape(),64,64,F=2, S=2, scope="unpool2")
        deconv2b = conv(unpool2, training,D=64, K=64, F=7, S=1, scope = "deconv2b")
        deconv2a = conv(deconv2b, training,D=64, K=64, F=7, S=1, scope = "deconv2a")

        unpool1 = unpool(deconv2a, conv1b.get_shape(),64,64,F=2, S=2, scope="unpool1")
        deconv1b = conv(unpool1, training,D=64, K=64, F=7, S=1, scope = "deconv1b")
        deconv1a = conv(deconv1b, training,D=64, K=64, F=7, S=1, scope = "deconv1a")

        logits = conv(deconv1a, training, D=64, K=11, F=1, S=1, scope="logits", activation=False)
        return logits

def inference(logits):
    return tf.argmax(tf.nn.softmax(logits, axis=3), axis=3)
