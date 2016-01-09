import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.getcwd()+'/YOLO.Tensorflow/')
from utils.PythonReader import ReadGoogleNetWeights
from PIL import Image
'''
"""
avg pooling experiment
"""
conv = tf.placeholder(tf.float32,[None,7,7,1024])

avg_pool = tf.nn.avg_pool(conv,ksize=[1,7,7,1],strides=[1,7,7,1],padding='SAME')

sess = tf.Session()
matrix = np.random.rand(7,7,1024)
for i in range(0,7):
    for j in range(0,7):
        print matrix[i][j][0]
matrix_reshaped = np.reshape(matrix,[-1,7,7,1024])


pooled_matrix = sess.run(avg_pool,feed_dict={conv:matrix_reshaped})
print pooled_matrix.shape
print pooled_matrix[0][0][0][0]

'''
'''
matrix = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
#x = tf.Variable(tf.random_normal([18]))

#x = tf.assign(x,matrix)
x = tf.Variable(matrix)
x = tf.reshape(x,[1,2,3,3])
y = 2*x

sess = tf.Session()
sess.run(tf.initialize_all_variables())
return_value = sess.run([x,y])  # or `assign_op.op.run()`
print return_value[0]
print '============='
print return_value[1]
'''

'''
googleNet = ReadGoogleNetWeights(os.getcwd()+'/YOLO.Tensorflow/extraction.weights')
print googleNet.layers[1].biases.shape
print googleNet.layers[1].weights.shape
x = tf.Variable(tf.random_normal([7*7*3*64]))
x = tf.assign(x,googleNet.layers[1].weights)
x = tf.reshape(x,[7,7,3,64])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
return_value = sess.run(x)
print return_value.shape
'''
'''
from scipy import misc

im = Image.open('eagle.jpg')
ImageSize = im.size
print im.size
WARP_LENGTH = 224
if(ImageSize[0] <= ImageSize[1]):
    width = int(ImageSize[1] * WARP_LENGTH / ImageSize[0])
    im = im.resize((WARP_LENGTH,width),Image.ANTIALIAS)
else:
    height = int(ImageSize[0] * WARP_LENGTH / ImageSize[1])
    im = im.resize((height,WARP_LENGTH),Image.ANTIALIAS)
im.save('eagle_resize.jpg')
img = misc.imread('eagle_resize.jpg')

img_tf = tf.Variable(img)
img_tf = tf.image.resize_image_with_crop_or_pad(img_tf, 224, 224)
img_tf = tf.cast(img_tf, tf.float32)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
im = sess.run(img_tf)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.add_subplot(1,2,1)

fig.add_subplot(1,2,2)
plt.imshow(img)
plt.show()
'''
def leaky_relu(m):
    index = m < 0
    m[index] = m[index] * 0.1
    return m
x = [1.0,2.2,3.0,-1.0,-2.0,1.0,1.1,-0.9,-0.8,1.0,2.2,3.0,-1.0,-2.0,1.0,1.1,-0.9,-0.8]
x = np.asarray(x)
x = np.reshape(x,(2,3,3))
print leaky_relu(x)
