import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model_t as net
# import os

batch_size = 1
epochs = 500
summaries_dir = './summaries'

image = (np.array(Image.open("./CamVid/train/0001TP_006690.png"))).astype('float32').reshape(1,360,480,3)
label = (np.array(Image.open("./CamVid/trainannot/0001TP_006690.png"))).astype('int').reshape(1,360,480)

# images = np.array(Image.open("./CamVid/train/0001TP_006690.png"))
# plt.imshow(images)
# # plt.imshow(label)
# plt.show()label

image_input = tf.placeholder(tf.float32, shape=(batch_size, 360, 480, 3), name="input")
label_input = tf.placeholder(tf.int32, shape=(batch_size, 360, 480), name="label")
# is_training =tf.placeholder(tf.bool, "is_training")
is_training = tf.placeholder_with_default(False, shape=(),name='is_training')

preds = net.predict(image_input, is_training)
loss = net.loss(preds, label_input)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
        train = tf.train.AdamOptimizer().minimize(loss)

merged = tf.summary.merge_all()

sess = tf.Session()
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(summaries_dir + '/test')

sess.run(tf.global_variables_initializer())

for step in range(epochs):
    _, _loss, summary = sess.run([train, loss, merged], feed_dict={image_input: image, label_input: label, is_training: True})
    if step % 10 == 0:
        train_writer.add_summary(summary, step)
        print("step= ",step)
        print("loss= ",_loss)

prediction = sess.run([preds], feed_dict={image_input: image, label_input: label, is_training: False})
