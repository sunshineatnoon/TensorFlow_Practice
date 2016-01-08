import tensorflow as tf
import numpy as np
import input_data

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
flags.DEFINE_integer('dropout',0.75,'The propobility to keep weights')

def input_placeholders(n_input,n_classes):
    """
    Returns:
      x: input images placeholders
      y: output labels placeholders
      keep_prob: keep probability placeholer
    """
    x = tf.placeholder(tf.float32,[None,n_input])
    y = tf.placeholder(tf.float32,[None,n_classes])
    keep_prob = tf.placeholder(tf.float32)
    return x,y,keep_prob

def paramters_variables(n_classes):
    """
    Returns:
      weights: weights variables of each layer
      biases: biases variables of each layer
    """
    weights = {
        'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
        'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
        'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
        'out':tf.Variable(tf.random_normal([1024,n_classes]))
    }
    biases = {
        'bc1':tf.Variable(tf.random_normal([32])),
        'bc2':tf.Variable(tf.random_normal([64])),
        'bd1':tf.Variable(tf.random_normal([1024])),
        'out':tf.Variable(tf.random_normal([n_classes]))
    }
    return weights,biases

def conv2d(img,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,1,1,1],padding='SAME'),b))

def max_pool(img,k):
    return tf.nn.max_pool(img,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def feed_forward(_X,_weights,_biases,_dropout):
    """
    Args:
      _X: batch of images placeholders
      _weights: weight variables
      _biases: biases variables
      _dropout: keep probability
    Return:
      out: Predictions of the forward pass
    """
    _X = tf.reshape(_X,[-1,28,28,1])

    #Convolutional Layer 1
    conv1 = conv2d(_X,_weights['wc1'],_biases['bc1'])
    conv1 = max_pool(conv1,k=2)
    conv1 = tf.nn.dropout(conv1,_dropout)

    #Convolutional Layer 2
    conv2 = conv2d(conv1,_weights['wc2'],_biases['bc2'])
    conv2 = max_pool(conv2,k=2)
    conv2 = tf.nn.dropout(conv2,_dropout)

    #Fully Connected Layer 1
    dense1 = tf.reshape(conv2,[-1,_weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1,_weights['wd1']),_biases['bd1']))
    dense1 = tf.nn.dropout(dense1,_dropout)

    #Prediction Layer
    out = tf.add(tf.matmul(dense1,_weights['out']),_biases['out'])
    return out

def loss_fun(pred,y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
    return loss

def optimizer_fun(learning_rate,loss):
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer

def evaluate(pred,y):
    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    return correct_pred,accuracy

mnist = input_data.read_data_sets(FLAGS.train_dir,one_hot=True)
#define graph
x,y,keep_prob = input_placeholders(FLAGS.n_input,FLAGS.n_classes)
weights,biases = paramters_variables(FLAGS.n_classes)
pred = feed_forward(x,weights,biases,keep_prob)
cost = loss_fun(pred,y)
optimizer = optimizer_fun(FLAGS.learning_rate,cost)
correct_pred,accuracy = evaluate(pred,y)

#initialization
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('./tensorflow_logs', graph_def=sess.graph_def)
    step = 1
    while step * FLAGS.batch_size < FLAGS.max_steps:
        batch_xs,batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,keep_prob:FLAGS.dropout})
        if step%FLAGS.display_frequency == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            loss = sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            print "Iter "+str(step*FLAGS.batch_size) + ", Minibatch Loss= "+"{:.6f}".format(loss)+ ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
