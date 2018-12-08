import tensorflow as tf
import numpy as np
from PIL import Image
import helper
# import tensorflow.contrib.slim as slim
# %matplotlib inline

import matplotlib.pyplot as plt

image = np.array(Image.open("./CamVid/train/0001TP_006690.png"))
label = np.array(Image.open("./CamVid/trainannot/0001TP_006690.png"))
plt.imshow(image)
plt.imshow(label)
plt.show()
