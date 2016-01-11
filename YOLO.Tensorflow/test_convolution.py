import tensorflow as tf
import numpy as np

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

def leaky_relu(m):
    return tf.maximum(0.1*m,m)

def conv2d(img,w,b,k):
    return leaky_relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,k,k,1],padding='SAME'),b))

data = np.asarray([1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3])

image = tf.Variable(data)
image = tf.reshape(image,[-1,5,5,3])
image = tf.cast(image,tf.float32)

dt = np.dtype("float32")
testArray = np.fromfile('tiny_work.weights',dtype=dt)

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
