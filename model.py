import tensorflow as tf
import numpy as np

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def conv(x, D, K, F, S, scope):
    with tf.variable_scope(scope):
        strides = [1, S, S, 1]
        with tf.name_scope('filter_weights'):
            filter = tf.get_variable('filter', [F,F,D,K],
             initializer = tf.truncated_normal_initializer(stddev=0.1))
            variable_summaries(filter)
        with tf.name_scope('bias'):
            bias = tf.get_variable('bias', [K], initializer = tf.constant_initializer(0.01))
            variable_summaries(bias)
        with tf.name_scope('convolved_output'):
            x = tf.nn.conv2d(x, filter, strides, padding = 'SAME', name=scope)
            tf.summary.histogram('pre-activations', x)
            x = tf.nn.relu(tf.nn.bias_add(x,bias))
            tf.summary.histogram('activations', x)
        return x

def max_pool(x, F, S, scope):
    with tf.name_scope(scope):
        strides = [1,S,S,1]
        op,indices = tf.nn.max_pool_with_argmax(x, [1, F, F, 1], strides, padding = 'SAME')
        tf.summary.histogram('max_pool_outputs', op)
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
        with tf.name_scope('filter'):
            filter = tf.get_variable('filter', [F,F,D,K],
             initializer = tf.truncated_normal_initializer(stddev=0.1))
            variable_summaries(filter)
        strides = [1,S,S,1]
        with tf.name_scope('unpool_outputs'):
            op = tf.nn.conv2d_transpose(x, filter, output_shape,strides,padding = 'SAME')
            tf.summary.histogram('unpool_outputs', op)
    return op

def loss(input_preds, input_labels):
    with tf.name_scope('one_hot_reshape'):
        one_hot_labels = tf.one_hot(input_labels, depth=12)
    with tf.name_scope('cross_entropy'):
        ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels,logits=input_preds, name='cross_entropy')
    # ce_loss = tf.softmax_cross_entropy_with_logits(labels=input_labels,logits=input_preds, name='cross_entropy')
        ce_mean = tf.reduce_mean(ce_loss, name='ce_mean')
        tf.summary.scalar('cross_entropy', ce_mean)
    return ce_mean

def predict(image):
    img = tf.nn.lrn(image, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm_img')

    with tf.variable_scope('Encoders'):
        conv1 = conv(img, D=3, K=64, F=7, S=1, scope = "conv1")
        pool1, pool1_indices = max_pool(conv1, F=2, S=2, scope = "pool1") #180*240

        conv2 = conv(pool1, D=64, K=64, F=7, S=1, scope = "conv2")
        pool2, pool2_indices = max_pool(conv2, F=2, S=2, scope = "pool2") #90*120

        conv3 = conv(pool2, D=64, K=64, F=7, S=1, scope = "conv3")
        pool3, pool3_indices = max_pool(conv3, F=2, S=2, scope = "pool3") #45*60

        conv4 = conv(pool3, D=64, K=64, F=7, S=1, scope = "conv4")
        pool4, pool4_indices = max_pool(conv4, F=2, S=2, scope = "pool4") #23*30

    with tf.variable_scope('Decoders'):
        unpool4 = unpool(pool4, conv4.get_shape(),64,64,2, S=2, scope="unpool4")
        # unpool4 = max_indices_unpool(pool4, pool4_indices, S=2, scope = "unpool4")
        deconv4 = conv(unpool4, D = 64, K=64, F=7, S=1, scope = "deconv4")

        unpool3 = unpool(pool3, conv3.get_shape(),64,64,2, S=2, scope="unpool3")
        # unpool3 = max_indices_unpool(deconv4, pool3_indices, S=2, scope = "unpool3")
        deconv3 = conv(unpool3, D=64, K=64, F=7, S=1, scope = "deconv3")

        unpool2 = unpool(pool2, conv2.get_shape(),64,64,2, S=2, scope="unpool2")
        # unpool2 = max_indices_unpool(deconv3, pool2_indices, S=2, scope = "unpool2")
        deconv2 = conv(unpool2, D=64, K=64, F=7, S=1, scope = "deconv2")

        unpool1 = unpool(pool1, conv1.get_shape(),64,64,2, S=2, scope="unpool1")
        # unpool1 = max_indices_unpool(deconv2, pool1_indices, S=2, scope = "unpool1")
        deconv1 = conv(unpool1, D=64, K=64, F=3, S=1, scope = "deconv1")

    preds = tf.nn.softmax(conv(deconv1, D=64, K=12, F=1, S=1, scope="classify"))
    # logits = tf.cast(tf.argmax(preds, axis=3), tf.int32)
    return  preds
