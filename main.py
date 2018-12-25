import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import model as net
import utilities as utils

"""
Runs tf sessions here
"""

cwd = os.getcwd()
batch_size = 1
epochs = 4
summaries_dir = './summaries'

def inputs(images, labels, is_training):
    """
    """
    train_image_directories = tf.data.Dataset.list_files(cwd + images)
    train_label_directories = tf.data.Dataset.list_files(cwd + labels)
    train_images = train_image_directories.map(lambda x: tf.image.decode_png(tf.read_file(x)))
    train_label = train_label_directories.map(lambda x: tf.image.decode_png(tf.read_file(x)))
    dataset = tf.data.Dataset.zip((train_images, train_label)).shuffle(30).batch(batch_size).prefetch(1)
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    init_op = iterator.initializer
    return [images, labels, init_op]

image_input = tf.placeholder(tf.float32, shape=(batch_size, 360, 480, 3), name="input")
label_input = tf.placeholder(tf.int32, shape=(batch_size, 360, 480,1), name="label")
is_training = tf.placeholder_with_default(False, shape=(),name='is_training')

logits = net.model(image_input, is_training)
loss = net.loss(logits, label_input)
# accuracy = net.accuracy(logits, label_input) 
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
        train = tf.train.AdamOptimizer().minimize(loss)
merged = tf.summary.merge_all()

sess = tf.Session()
input = inputs(images = '/CamVid/train/*.png', labels = '/CamVid/trainannot/*.png', is_training=True)
sess.run(input[2])
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(summaries_dir + '/test')

image_t = (np.array(Image.open("./CamVid/train/0001TP_006690.png"))).astype('float32').reshape(1,360,480,3)
labels_t = (np.array(Image.open("./CamVid/trainannot/0001TP_006690.png"))).astype('int').reshape(360,480)

# print(labels_t)
# img = utils.showLabel(labels_t)
# plt.imshow(img)
# plt.show()

print('Training:')
for step in range(epochs):
    try:
        image, labels = sess.run([input[0], input[1]])
        _, _loss, summary = sess.run([train, loss, merged], feed_dict={image_input: image, label_input: labels, is_training: True})
        if step % 1 == 0:
            train_writer.add_summary(summary, step)
            print("step= ",step)
            print("loss= ",_loss)

    except tf.errors.OutOfRangeError:
        print("End of training dataset")
        break

print('Training completed')

preds = net.inference(image_input, is_training)
pred = sess.run([preds], feed_dict={image_input : image, is_training: True})
print(preds)
# utils.showLabel(preds)
