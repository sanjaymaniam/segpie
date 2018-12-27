import tensorflow as tf
from PIL import Image
import numpy as np
import os

import model as net
import utilities as utils

#Directories
cwd = os.getcwd()
summaries_dir = './summaries'
train_img_dir = '/CamVid/train/*.png'
train_annot_dir = '/CamVid/trainannot/*.png'
val_img_dir = '/CamVid/train/*.png'
val_annot_dir = '/CamVid/trainannot/*.png'
test_img_dir = '/CamVid/train/*.png'
test_annot_dir = '/CamVid/trainannot/*.png'

#Hyperparameters
batch_size = 1
train_epochs = 200

#Placeholders
image_input = tf.placeholder(tf.float32, shape=(batch_size, 360, 480, 3), name="input")
label_input = tf.placeholder(tf.int32, shape=(batch_size, 360, 480,1), name="label")
is_training = tf.placeholder_with_default(False, shape=(),name='is_training')

#Iterators to load data
train_input = utils.inputs(cwd, train_img_dir, train_annot_dir, batch_size)
val_input = utils.inputs(cwd, val_img_dir, val_annot_dir, batch_size)
test_input = utils.inputs(cwd, test_img_dir, test_annot_dir, batch_size)

#Fetch nodes
logits = net.predict(image_input, is_training)
loss = net.loss(logits, label_input)
preds = net.inference(logits)
accuracy = net.accuracy(preds, label_input)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer().minimize(loss)

#Merge all variable summaries for tensorboard
merged = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#FileWriters for tensorboard
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
val_writer = tf.summary.FileWriter(summaries_dir + '/val')
test_writer = tf.summary.FileWriter(summaries_dir + '/test')

#Training phase:
sess.run(train_input[2])
print('Training...')
for step in range(train_epochs):
    try:
        image, labels = sess.run([train_input[0], train_input[1]])
        _, _loss, summary = sess.run([train, loss, merged], feed_dict={image_input: image, label_input: labels, is_training: True})
        if step % 1 == 0:
            train_writer.add_summary(summary, step)
            print("step= ",step)
            print("loss= ",_loss)
    except tf.errors.OutOfRangeError:
        print("End of training dataset")
        break
print('Training completed')

#Trial
image_t = (np.array(Image.open("./CamVid/train/0001TP_006690.png"))).astype('float32').reshape(1,360,480,3)
labels_t = (np.array(Image.open("./CamVid/trainannot/0001TP_006690.png"))).astype('int').reshape(360,480)
prediction = sess.run([preds], feed_dict={image_input: image_t})
X= np.array(prediction)
utils.show(X)
