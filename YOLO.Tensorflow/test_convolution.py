import tensorflow as tf
import numpy as np
from utils.PythonReader import ReadGoogleNetWeights
import os

'''
h=5
w=5
c=3
n=2
size=5
stride = 2
pad = 1

def assign_weights_func(tf_weights,weights):
    tf_weights = tf.assign(tf_weights,weights)
    tf_weights = tf.reshape(tf_weights,[2,3,5,5])
    tf_weights = tf.transpose(tf_weights,[3,2,1,0])
    return tf_weights

def assign_biases_func(tf_biases,biases):
    tf_biases = tf.assign(tf_biases,biases)
    return tf_biases





data = np.asarray([[1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1],
        [2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2],
        [3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3]])


data = np.asarray(
       [0,0,0,0,0,
        0,0,0,0,0,
        0,0,1,1,1,
        0,0,1,1,1,
        0,0,1,1,1,
        0,0,0,0,0,
        0,0,0,0,0,
        0,0,2,2,2,
        0,0,2,2,2,
        0,0,2,2,2,
        0,0,0,0,0,
        0,0,0,0,0,
        0,0,3,3,3,
        0,0,3,3,3,
        0,0,3,3,3])

data = np.reshape(data,[1,3,5,5])

image = tf.Variable(data)
image = tf.cast(image,tf.float32)
image = tf.transpose(image,[0,3,2,1])

dt = np.dtype("float32")
testArray = np.fromfile('tiny_work.weights',dtype=dt)
#weight_array = np.reshape(testArray[2:152],[2,3,5,5])
tf_weights = tf.Variable(testArray[2:152])
tf_weights = tf.reshape(tf_weights,[2,3,5,5])
tf_weights = tf.transpose(tf_weights,[3,2,1,0])

tf_biases = tf.Variable(tf.zeros([n]))

conv_feature = conv2d(image,tf_weights,tf_biases,2)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
feature = sess.run(conv_feature)

print feature.shape
print feature

tf_weights = tf.Variable(tf.random_normal([n*c*size*size]))
tf_weights = assign_weights_func(tf_weights,testArray[2:152])

tf_biases = tf.Variable(tf.random_normal([n]))
tf_biases = assign_biases_func(tf_biases,np.asarray(testArray[0:2]))

conv_feature = conv2d(image,tf_weights,tf_biases,2)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
feautre = sess.run(conv_feature)

print feautre.shape
print feautre

def leaky_relu(m):
    return tf.maximum(0.1*m,m)

def conv2d(img,w,b,k):
    return leaky_relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,k,k,1],padding='SAME'),b))

img = np.zeros([224,224,3])
img = img - 1
tf_img = tf.Variable(img)
tf_img = tf.reshape(tf_img,[-1,224,224,3])
tf_img = tf.cast(tf_img,tf.float32)

googleNet = ReadGoogleNetWeights(os.path.join(os.getcwd(),'extraction.weights'))
weight_array = googleNet.layers[1].weights
tf_weights = tf.Variable(weight_array)
tf_weights = tf.reshape(tf_weights,[64,3,7,7])
tf_weights = tf.transpose(tf_weights,[3,2,1,0])

tf_biases = tf.Variable(googleNet.layers[1].biases)

conv_feautre = conv2d(tf_img,tf_weights,tf_biases,2)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
feautre = sess.run(conv_feautre)

print feautre.shape
print feautre[0,0,0,0]

data = np.zeros(147)
data = data - 1

print np.sum(weight_array[0:147]*data)+googleNet.layers[1].biases[0]
'''
def im2col(image,ksize,stride):
    """
    Args:
      image:input image,shape should be like [h,w]
      ksize:weight size of kernels
      stride:stride when make convolutions
    Returns:
      matrix: a 2-D matrix of size (out_h*out*w)*(ksize*ksize),each row is a patch of the image of size 7*7
    """
    [h,w] = image.shape
    out_h = (h-ksize)/stride+1
    out_w = (w-ksize)/stride+1
    matrix = np.zeros([out_h*out_w,ksize*ksize])
    for i in range(out_h*out_w):
        #for each patch
        row = i / out_w
        col = i % out_w
        h_start = stride*row
        w_start = stride*col
        patch = image[h_start:h_start+ksize,w_start:w_start+ksize]
        patch = np.reshape(patch,[-1])
        matrix[i,:] = patch
    return matrix

def im_pad(image,padding_size):
    """
    Args:
      image: input image to be padded
      padding_size: padding sizes
    Returns:
      A padded image of size [h+2*padding_size,w+2*padding_size]
    """
    [h,w] = image.shape
    padded_image = np.zeros([h+2*padding_size,w+2*padding_size])
    padded_image[padding_size:padding_size+h,padding_size:padding_size+w] = image
    return padded_image


def convol(images,weights,biases,stride):
    """
    Args:
      images:input images or features, 4-D tensor
      weights:weights, 4-D tensor
      biases:biases, 1-D tensor
      stride:stride, a float number
    Returns:
      conv_feature: convolved feature map
    """
    image_num = images.shape[0] #the number of input images or feature maps
    channel = images.shape[1] #channels of an image,images's shape should be like [n,c,h,w]
    weight_num = weights.shape[0] #number of weights, weights' shape should be like [n,c,size,size]
    ksize = weights.shape[2]
    h = images.shape[2]
    w = images.shape[3]
    out_h = (h+np.floor(ksize/2)*2-ksize)/2+1
    out_w = out_h

    conv_features = np.zeros([image_num,weight_num,out_h,out_w])
    for i in range(image_num):
        image = images[i,...,...,...]
        for j in range(weight_num):
            sum_convol_feature = np.zeros([out_h,out_w])
            for c in range(channel):
                #extract a single channel image
                channel_image = image[c,...,...]
                #pad the image
                padded_image = im_pad(channel_image,ksize/2)
                #transform this image to a vector
                im_col = im2col(padded_image,ksize,stride)

                weight = weights[j,c,...,...]
                weight_col = np.reshape(weight,[-1])
                mul = np.dot(im_col,weight_col)
                convol_feature = np.reshape(mul,[out_h,out_w])
                sum_convol_feature = sum_convol_feature + convol_feature
            conv_features[i,j,...,...] = sum_convol_feature + biases[j]
    return conv_features
    
googleNet = ReadGoogleNetWeights(os.path.join(os.getcwd(),'extraction.weights'))

img = np.zeros([1,3,224,224])
img = img - 1
img = np.rollaxis(img, 1, 4)

weight_array = googleNet.layers[1].weights
weight_array = np.reshape(weight_array,[64,3,7,7])

print weight_array.shape

biases_array = googleNet.layers[1].biases
#print convol(img,weight_array,biases_array,2)

tf_weight = tf.Variable(weight_array)

tf_img = tf.Variable(img)
tf_img = tf.cast(tf_img,tf.float32)

tf_biases = tf.Variable(biases_array)

conv_feature = tf.nn.bias_add(tf.nn.conv2d(tf_img,tf_weight,strides=[1,2,2,1],padding='SAME'),tf_biases)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
feautre = sess.run(conv_feature)

print feautre
