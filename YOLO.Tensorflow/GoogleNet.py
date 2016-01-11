import tensorflow as tf
import numpy as np
import os,sys
import collections
sys.path.append(os.getcwd())
from utils.PythonReader import ReadGoogleNetWeights
from PIL import Image
from scipy import misc

#hyperparameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate',0.01,'Initial learning rate')
flags.DEFINE_integer('max_steps',100000,'Number of steps to run trainer')
flags.DEFINE_integer('batch_size',128,'Batch Size')
flags.DEFINE_string('train_dir','./data','Directory to put the training data')
flags.DEFINE_integer('display_frequency',10,'Display cost every x steps')
flags.DEFINE_integer('n_classes',10,'Number of output classes')
flags.DEFINE_integer('n_input',784,'Size of input vector')
flags.DEFINE_float('dropout',0.75,'The propobility to keep weights')

flags.DEFINE_integer('height',224,'Height of input images')
flags.DEFINE_integer('width',224,'Width of input images')
flags.DEFINE_integer('channels',3,'Channel number of input vector')

def crop(m,scale=2.0):
    trans = np.ones((1,224,224,3))
    return tf.add(tf.mul(m,scale),-1.0*trans)

def leaky_relu(m):
    return tf.maximum(0.1*m,m)

def input_placeholders(height,width,channels,n_classes):
    """
    Returns:
      x: input images placeholders
      y: output labels placeholders
    """
    x = tf.placeholder(tf.float32,[None,height,width,channels])
    y = tf.placeholder(tf.float32,[None,n_classes])
    return x,y

def paramters_variables(n_classes):
    """
    Returns:
      weights: weights variables of each layer
      biases: biases variables of each layer
    """
    weights = {
        'wc2':tf.Variable(tf.random_normal([7*7*3*64])),
        'wc4':tf.Variable(tf.random_normal([3*3*64*192])),
        'wc6':tf.Variable(tf.random_normal([1*1*192*128])),
        'wc7':tf.Variable(tf.random_normal([3*3*128*256])),
        'wc8':tf.Variable(tf.random_normal([1*1*256*256])),
        'wc9':tf.Variable(tf.random_normal([3*3*256*512])),
        'wc11':tf.Variable(tf.random_normal([1*1*512*256])),
        'wc12':tf.Variable(tf.random_normal([3*3*256*512])),
        'wc13':tf.Variable(tf.random_normal([1*1*512*256])),
        'wc14':tf.Variable(tf.random_normal([3*3*256*512])),
        'wc15':tf.Variable(tf.random_normal([1*1*512*256])),
        'wc16':tf.Variable(tf.random_normal([3*3*256*512])),
        'wc17':tf.Variable(tf.random_normal([1*1*512*256])),
        'wc18':tf.Variable(tf.random_normal([3*3*256*512])),
        'wc19':tf.Variable(tf.random_normal([1*1*512*512])),
        'wc20':tf.Variable(tf.random_normal([3*3*512*1024])),
        'wc22':tf.Variable(tf.random_normal([1*1*1024*512])),
        'wc23':tf.Variable(tf.random_normal([3*3*512*1024])),
        'wc24':tf.Variable(tf.random_normal([1*1*1024*512])),
        'wc25':tf.Variable(tf.random_normal([3*3*512*1024])),
        'wd27':tf.Variable(tf.random_normal([1024*1000])),
    }
    biases = {
        'bc2':tf.Variable(tf.random_normal([64])),
        'bc4':tf.Variable(tf.random_normal([192])),
        'bc6':tf.Variable(tf.random_normal([128])),
        'bc7':tf.Variable(tf.random_normal([256])),
        'bc8':tf.Variable(tf.random_normal([256])),
        'bc9':tf.Variable(tf.random_normal([512])),
        'bc11':tf.Variable(tf.random_normal([256])),
        'bc12':tf.Variable(tf.random_normal([512])),
        'bc13':tf.Variable(tf.random_normal([256])),
        'bc14':tf.Variable(tf.random_normal([512])),
        'bc15':tf.Variable(tf.random_normal([256])),
        'bc16':tf.Variable(tf.random_normal([512])),
        'bc17':tf.Variable(tf.random_normal([256])),
        'bc18':tf.Variable(tf.random_normal([512])),
        'bc19':tf.Variable(tf.random_normal([512])),
        'bc20':tf.Variable(tf.random_normal([1024])),
        'bc22':tf.Variable(tf.random_normal([512])),
        'bc23':tf.Variable(tf.random_normal([1024])),
        'bc24':tf.Variable(tf.random_normal([512])),
        'bc25':tf.Variable(tf.random_normal([1024])),
        'bd27':tf.Variable(tf.random_normal([1000])),
    }
    return weights,biases

def conv2d(img,w,b,k):
    return leaky_relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,k,k,1],padding='SAME'),b))

def max_pool(img,k):
    return tf.nn.max_pool(img,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def avg_pool(img,k):
    return tf.nn.avg_pool(img,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def feed_forward(_X,_weights,_biases):
    """
    Args:
      _X: batch of images placeholders
      _weights: weight variables
      _biases: biases variables
    Return:
      out: Predictions of the forward pass
    """
    _X = tf.cast(_X,tf.float32)
    _X = tf.reshape(_X,[-1,224,224,3])
    cropped = crop(_X)

    #Convolutional Layer 2, CROP is the first layer
    conv2 = conv2d(cropped,_weights['wc2'],_biases['bc2'],2)

    conv2_pooled = max_pool(conv2,k=2)

    #Convolutional Layer 4
    conv4 = conv2d(conv2_pooled,_weights['wc4'],_biases['bc4'],1)
    conv4 = max_pool(conv4,k=2)

    #Convolutional Layer 6~9
    conv6 = conv2d(conv4,_weights['wc6'],_biases['bc6'],1)
    conv7 = conv2d(conv6,_weights['wc7'],_biases['bc7'],1)
    conv8 = conv2d(conv7,_weights['wc8'],_biases['bc8'],1)
    conv9 = conv2d(conv8,_weights['wc9'],_biases['bc9'],1)
    conv9 = max_pool(conv9,k=2)

    #Convolutional Layer 11~20
    conv11 = conv2d(conv9,_weights['wc11'],_biases['bc11'],1)
    conv12 = conv2d(conv11,_weights['wc12'],_biases['bc12'],1)
    conv13 = conv2d(conv12,_weights['wc13'],_biases['bc13'],1)
    conv14 = conv2d(conv13,_weights['wc14'],_biases['bc14'],1)
    conv15 = conv2d(conv14,_weights['wc15'],_biases['bc15'],1)
    conv16 = conv2d(conv15,_weights['wc16'],_biases['bc16'],1)
    conv17 = conv2d(conv16,_weights['wc17'],_biases['bc17'],1)
    conv18 = conv2d(conv17,_weights['wc18'],_biases['bc18'],1)
    conv19 = conv2d(conv18,_weights['wc19'],_biases['bc19'],1)
    conv20 = conv2d(conv19,_weights['wc20'],_biases['bc20'],1)
    conv20 = max_pool(conv20,k=2)

    #Convolutional Layer 22~25
    conv22 = conv2d(conv20,_weights['wc22'],_biases['bc22'],1)
    conv23 = conv2d(conv22,_weights['wc23'],_biases['bc23'],1)
    conv24 = conv2d(conv23,_weights['wc24'],_biases['bc24'],1)
    conv25 = conv2d(conv24,_weights['wc25'],_biases['bc25'],1)
    conv25 = tf.nn.avg_pool(conv25,ksize=[1,7,7,1],strides=[1,7,7,1],padding='SAME')

    #Fully Connected Layer 1
    dense1 = tf.reshape(conv25,[-1,_weights['wd27'].get_shape().as_list()[0]])
    dense1 = leaky_relu(tf.add(tf.matmul(dense1,_weights['wd27']),_biases['bd27']))

    return dense1,conv2

def assign_weights_func(layer_number,weights,layer_name):
    l = googleNet.layers[layer_number]
    weights[layer_name] = tf.assign(weights[layer_name],l.weights)
    if(l.type == "CONVOLUTIONAL"):
        #weights[layer_name] = tf.reshape(weights[layer_name],[l.n,l.c,l.size,l.size])
        #weights[layer_name] = tf.transpose(weights[layer_name],[2,3,1,0])
        weights[layer_name] = tf.reshape(weights[layer_name],[l.n,l.size,l.size,l.c])
        weights[layer_name] = tf.transpose(weights[layer_name],[1,2,3,0])
    elif(l.type == "CONNECTED"):
        weights[layer_name] = tf.reshape(weights[layer_name],[l.input_size,l.output_size])
    return weights[layer_name]

def assign_biases_func(layer_number,biases,layer_name):
    l = googleNet.layers[layer_number]
    biases[layer_name] = tf.assign(biases[layer_name],l.biases)
    return biases[layer_name]

#read an image

im = Image.open('/home/xuetingli/Documents/YOLO.Tensorflow/TensorFlow_Practice/outfile.jpg')
ImageSize = im.size
print im.size
WARP_LENGTH = 224
if(ImageSize[0] <= ImageSize[1]):
    width = int(ImageSize[1] * WARP_LENGTH / ImageSize[0])
    im = im.resize((WARP_LENGTH,width),Image.ANTIALIAS)
    imSize = im.size
    print "the image size is ",imSize[1]
    im = im.crop((0,imSize[1]/2-112,224,imSize[1]/2+112))
else:
    height = int(ImageSize[0] * WARP_LENGTH / ImageSize[1])
    im = im.resize((height,WARP_LENGTH),Image.ANTIALIAS)
    imSize = im.size
    print "the image size is ",imSize[0]
    im = im.crop((imSize[0]/2-112,0,imSize[0]/2+112,224))

im.save('eagle_resize.jpg')

img = misc.imread('eagle_resize.jpg')
img_tf = tf.Variable(img/255.0)
#img_tf = tf.image.resize_image_with_crop_or_pad(img_tf, 224, 224)
#img_tf = tf.cast(img_tf, tf.float32)

'''
img = np.zeros((224,224,3))
img_tf = tf.Variable(img)
import scipy
scipy.misc.imsave('outfile.jpg', img)
'''
#Read and assign weights
weights,biases = paramters_variables(FLAGS.n_classes)
googleNet = ReadGoogleNetWeights(os.path.join(os.getcwd(),'extraction.weights'))

for i in range(googleNet.layer_number):
    l = googleNet.layers[i]
    if(l.type == "CONVOLUTIONAL"):
        weights['wc'+str(i+1)] = assign_weights_func(i,weights,'wc'+str(i+1))
        biases['wd'+str(i+1)] = assign_biases_func(i,biases,'bc'+str(i+1))
    elif(l.type == "CONNECTED"):
        weights['wc'+str(i+1)] = assign_weights_func(i,weights,'wd'+str(i+1))
        biases['wd'+str(i+1)] = assign_biases_func(i,biases,'bd'+str(i+1))

#feed forward process
[out,conv2] = feed_forward(img_tf,weights,biases)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

four = sess.run([out,conv2,weights['wc2'],weights['wc4'],weights['wc6'],weights['wc7'],weights['wc8'],weights['wc9'],weights['wc11'],
                weights['wc12'],weights['wc13'],weights['wc14'],weights['wc15'],weights['wc16'],weights['wc17'],weights['wc18'],
                weights['wc19'],weights['wc20'],weights['wc22'],weights['wc23'],weights['wc24'],weights['wc25'],weights['wd27'],
                biases['bc2'],biases['bc4'],biases['bc6'],biases['bc7'],biases['bc8'],biases['bc9'],biases['bc11'],
                biases['bc12'],biases['bc13'],biases['bc14'],biases['bc15'],biases['bc16'],biases['bc17'],biases['bc18'],
                biases['bc19'],biases['bc20'],biases['bc22'],biases['bc23'],biases['bc24'],biases['bc25'],biases['bd27'],img_tf])

#four = sess.run([cropped,conv2,weights['wc2'],biases['bc2'],img_tf])

print four[1]
print np.argmax(four[0])
#print four[42]
print four[len(four)-1][...,...,0]


'''
import matplotlib.pyplot as plt
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img)
fig.add_subplot(1,2,2)
plt.imshow(four[44])
plt.show()
'''
