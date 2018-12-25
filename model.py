import tensorflow as tf
import numpy as np

"""
Main graph nodes are defined here
"""

def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def conv(x, trainingstat, D, K, F, S, scope, activation=True):
    with tf.variable_scope(scope):
        strides = [1, S, S, 1]
        with tf.variable_scope('filter', reuse=tf.AUTO_REUSE):
            filter = tf.get_variable('filter', [F,F,D,K],
             initializer = tf.truncated_normal_initializer(stddev=0.01))
            variable_summaries(filter)
        with tf.variable_scope('bias', reuse=tf.AUTO_REUSE):
            bias = tf.get_variable('bias', [K], initializer = tf.constant_initializer(0))
            # variable_summaries(bias)
        x = tf.nn.bias_add(tf.nn.conv2d(x, filter, strides, padding = 'SAME', name=scope), bias)
        with tf.variable_scope('batch_norm'):
            x = tf.layers.batch_normalization(x, training=trainingstat)
            tf.summary.histogram("batch_norm",x)
        if(activation):
            x = tf.nn.relu(x)
        tf.summary.histogram(scope, x)
        return x

def max_pool(x, F, S, scope):
    with tf.variable_scope(scope):
        strides = [1,S,S,1]
        op,indices = tf.nn.max_pool_with_argmax(x, [1, F, F, 1], strides, padding = 'SAME')
        tf.summary.histogram(scope, op)
        return op, indices

"""
def max_indices_unpool(pool, ind, S=2 , scope='unpool_2d'):
  stride=[1, S, S, 1]

  with tf.variable_scope(scope):
    input_shape = tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                      shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret
"""

def unpool(x, output_shape,D,K,F, S, scope): #transpose convolution
    with tf.variable_scope(scope):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            filter = tf.get_variable('filter', [F,F,D,K],
             initializer = tf.truncated_normal_initializer(stddev=0.01))
            variable_summaries(filter)
        strides = [1,S,S,1]
        op = tf.nn.conv2d_transpose(x, filter, output_shape,strides,padding = 'SAME')
        tf.summary.histogram(scope, op)
        return op

def loss(logits, labels):
    one_hot_labels = tf.one_hot(labels, depth=11)
    with tf.variable_scope("cross_entropy"):
        ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels,logits=logits, name='cross_entropy')
        # ce_loss = tf.softmax_cross_entropy_with_logits(labels=input_labels,logits=input_preds, name='cross_entropy')
        ce_mean = tf.reduce_mean(ce_loss, name='ce_mean')
        tf.summary.scalar('cross_entropy', ce_mean)
        return ce_mean

# def accuracy(logits, labels):

def model(image, training):
    with tf.variable_scope('Model'):
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
        #unpool4 = max_indices_unpool(pool4, pool4_indices, S=2, scope = "unpool4")
        deconv4b = conv(unpool4, training, D = 64, K=64, F=7, S=1, scope = "deconv4b")
        deconv4a = conv(deconv4b, training, D = 64, K=64, F=7, S=1, scope = "deconv4a")

        unpool3 = unpool(deconv4a, conv3b.get_shape(),64,64,F=2, S=2, scope="unpool3")
        #unpool3 = max_indices_unpool(deconv4, pool3_indices, S=2, scope = "unpool3")
        deconv3b = conv(unpool3, training, D=64, K=64, F=7, S=1, scope = "deconv3b")
        deconv3a = conv(deconv3b, training, D=64, K=64, F=7, S=1, scope = "deconv3a")

        unpool2 = unpool(deconv3a, conv2b.get_shape(),64,64,F=2, S=2, scope="unpool2")
        #unpool2 = max_indices_unpool(deconv3, pool2_indices, S=2, scope = "unpool2")
        deconv2b = conv(unpool2, training,D=64, K=64, F=7, S=1, scope = "deconv2b")
        deconv2a = conv(deconv2b, training,D=64, K=64, F=7, S=1, scope = "deconv2a")

        unpool1 = unpool(deconv2a, conv1b.get_shape(),64,64,F=2, S=2, scope="unpool1")
        #unpool1 = max_indices_unpool(deconv2, pool1_indices, S=2, scope = "unpool1")
        deconv1b = conv(unpool1, training,D=64, K=64, F=7, S=1, scope = "deconv1b")
        deconv1a = conv(deconv1b, training,D=64, K=64, F=7, S=1, scope = "deconv1a")

        logits = conv(deconv1a, training, D=64, K=11, F=1, S=1, scope="logits", activation=False)
        return logits

def inference(image, training):
    preds = model(image, training)
    preds = tf.nn.softmax(preds, axis=3, name="Softmax_op")
    preds = tf.argmax(preds, axis=3)
    return preds
