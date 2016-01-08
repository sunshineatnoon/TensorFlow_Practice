import tensorflow as tf
import numpy as np

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
        'wc2':tf.Variable(tf.random_normal([7,7,3,64])),
        'wc4':tf.Variable(tf.random_normal([3,3,64,192])),
        'wc6':tf.Variable(tf.random_normal([1,1,192,128])),
        'wc7':tf.Variable(tf.random_normal([3,3,128,256])),
        'wc8':tf.Variable(tf.random_normal([1,1,256,256])),
        'wc9':tf.Variable(tf.random_normal([3,3,256,512])),
        'wc11':tf.Variable(tf.random_normal([1,1,512,256])),
        'wc12':tf.Variable(tf.random_normal([3,3,256,512])),
        'wc13':tf.Variable(tf.random_normal([1,1,512,256])),
        'wc14':tf.Variable(tf.random_normal([3,3,256,512])),
        'wc15':tf.Variable(tf.random_normal([1,1,512,256])),
        'wc16':tf.Variable(tf.random_normal([3,3,256,512])),
        'wc17':tf.Variable(tf.random_normal([1,1,512,256])),
        'wc18':tf.Variable(tf.random_normal([3,3,256,512])),
        'wc19':tf.Variable(tf.random_normal([1,1,512,512])),
        'wc20':tf.Variable(tf.random_normal([3,3,512,1024])),
        'wc22':tf.Variable(tf.random_normal([1,1,1024,512])),
        'wc23':tf.Variable(tf.random_normal([3,3,512,1024])),
        'wc24':tf.Variable(tf.random_normal([1,1,1024,512])),
        'wc25':tf.Variable(tf.random_normal([3,3,512,1024])),
        'wd27':tf.Variable(tf.random_normal([1024,1000])),
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
        'bc14':tf.Variable(tf.random_normal([512)),
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
cost_summ = tf.scalar_summary("loss summary",cost)
optimizer = optimizer_fun(FLAGS.learning_rate,cost)
correct_pred,accuracy = evaluate(pred,y)
accuracy_summary = tf.scalar_summary("accuracy",accuracy)

#initialization
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('./tensorflow_logs', graph_def=sess.graph_def)
    step = 1
    while step * FLAGS.batch_size < FLAGS.max_steps:
        batch_xs,batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,keep_prob:FLAGS.dropout})
        if step%FLAGS.display_frequency == 0:
            result = sess.run([merged,accuracy],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            summary_str = result[0]
            acc = result[1]
            summary_writer.add_summary(summary_str,step*FLAGS.batch_size)
            loss = sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            print "Iter "+str(step*FLAGS.batch_size) + ", Minibatch Loss= "+"{:.6f}".format(loss)+ ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
