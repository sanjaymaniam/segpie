import tensorflow as tf

 upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 45, 60, 64], 2, "up4")

def conv(x, D, K, F, S):
    strides = [1, S, S, 1]
    filter = tf.get_variables('filter', [F,F,D,K],
             initializer = tf.truncated_normal_initializer(stddev=0.1))
    bias = tf.get_variables('bias', [K], initializer = tf.constant_initializer(0.1))
    x = tf.nn.conv2d(x, filter, strides, padding = 'SAME')
    return tf.nn.relu(tf.nn.bias_add(x,bias))

def max_pool(x, F, S):
    strides = [1,S,S,1]
    return tf.nn.max_pool_with_argmax(x, [1, F, F, 1], strides, padding = 'SAME')

def transpose_conv(x, D, K, F, S):
    strides = [1,S, S, 1]
    filter = tf.get_variables('filter', [F,F,D,K],
             initializer = tf.truncated_normal_initializer(stddev=0.1))
    output_shape =     
    x = tf.nnconv2d_transpose(x, filter, output_shape, strides, padding = 'SAME')

def inference(images, labels, batch_size):
    img = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm_img')

    with tf.variable_scope('Encoders'):
        conv1 = conv(img, D=3, K=64, F=3, S=1)
        conv2 = conv(conv1, D=64, K=64, F=3, S=1)
        pool1, pool1_indices = max_pool(conv2, F=2, S=2)

        conv3 = conv(pool1, D=64, K=128, F=3, S=1)
        conv4 = conv(conv3, D=128, K=128, F=3, S=1)
        pool2, pool2_indices = max_pool(conv4, F=2, S=2)

        conv5 = conv(pool2, D=128, K=256, F=3, S=1)
        conv6 = conv(conv5, D=256, K=256, F=3, S=1)
        pool3, pool2_indices = max_pool(conv6, F=2, S=2)

        conv7 = conv(pool3, D=256, K=512, F=3, S=1)
        conv8 = conv(conv7, D=512, K=512, F=3, S=1)
        pool4, pool4_indices = max_pool(conv8, F=2, S=2)

    with tf.variable_scope('Decoders'):
        unpool4 = transpose_conv()
