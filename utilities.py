import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Misc supporting utils
"""

def inputs(cwd, images, labels, batch_size):
    train_image_directories = tf.data.Dataset.list_files(cwd + images)
    train_label_directories = tf.data.Dataset.list_files(cwd + labels)
    train_images = train_image_directories.map(lambda x: tf.image.decode_png(tf.read_file(x)))
    train_label = train_label_directories.map(lambda x: tf.image.decode_png(tf.read_file(x)))
    dataset = tf.data.Dataset.zip((train_images, train_label)).shuffle(30).batch(batch_size).prefetch(1)
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    init_op = iterator.initializer
    return [images, labels, init_op]

def show(image):
    """
    Reference: https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Scripts/test_segmentation_camvid.py
    """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0, 11):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((360, 480, 3))
    print(rgb.shape, r.shape)
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    plt.imshow(np.uint8(rgb))
    plt.show()
