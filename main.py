import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model as net
import os

BATCH_SIZE = 1
image = (np.array(Image.open("./CamVid/train/0001TP_006690.png"))).astype('float32').reshape(1,360,480,3)
label = (np.array(Image.open("./CamVid/trainannot/0001TP_006690.png"))).astype('int').reshape(1,360,480)

# images = np.array(Image.open("./CamVid/train/0001TP_006690.png"))
# plt.imshow(images)
# # plt.imshow(label)
# plt.show()label

image_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 360, 480, 3), name="input")
label_input = tf.placeholder(tf.int32, shape=(BATCH_SIZE, 360, 480), name="label")
# is_training =tf.placeholder(tf.bool, "is_training")

with tf.name_scope("feed_forward"):
    preds = net.predict(image_input)
with tf.name_scope("loss"):
    loss = net.loss(preds, label_input)
with tf.name_scope("train"):
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

"""
merged = tf.summary.merge_all()
if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','first')):
    os.mkdir(os.path.join('summaries','first'))
"""

sess = tf.Session()
# summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), sess.graph)
sess.run(tf.global_variables_initializer())
for step in range(500):
    _, _loss = sess.run([train, loss], feed_dict={image_input: image, label_input: label})
    if step % 10 == 0:
        print("step= ",step)
        print("loss= ",_loss)
