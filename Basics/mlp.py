import tensorflow as tf
import numpy as np
import math
import input_data
import time

#hyperparameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate',0.01,'Initial learning rate')
flags.DEFINE_integer('max_steps',5000,'Number of steps to run trainer')
flags.DEFINE_integer('hidden1',128,'Number of units in hidden layer 1')
flags.DEFINE_integer('hidden2',32,'Number of units in hidden layer 2')
flags.DEFINE_integer('batch_size',100,'Batch Size')
flags.DEFINE_string('train_dir','./data','Directory to put the training data')
flags.DEFINE_integer('image_pixels',28*28,'Batch Size')
flags.DEFINE_integer('num_classes',10,'Batch Size')


def placeholder_inputs(batch_size):
    """
    Args: Batch Size
    Returns:
      images_placeholders: Images placeholder
      labels_placeholders: Labels placeholder
    """
    images_placeholder = tf.placeholder(tf.float32,shape=(batch_size,FLAGS.image_pixels))
    labels_placeholder = tf.placeholder(tf.int32,shape=(batch_size))
    return images_placeholder,labels_placeholder

def Inference(images):
    """
    Args:
      images: image placeholders
    Returns:
      logits: Output tensor from the softmax linear layer
    """
    #Hidden Layer 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([FLAGS.image_pixels,FLAGS.hidden1],stddev=1.0/math.sqrt(float(FLAGS.image_pixels))),name="weights")
        biases = tf.Variable(tf.zeros([FLAGS.hidden1]),name="biases")
        hidden1 = tf.nn.relu(tf.matmul(images,weights)+biases)

    #Hidden Layer 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([FLAGS.hidden1,FLAGS.hidden2],stddev=1.0/math.sqrt(float(FLAGS.hidden1))),name="weights")
        biases = tf.Variable(tf.zeros([FLAGS.hidden2]),name="biases")
        hidden2 = tf.nn.relu(tf.matmul(hidden1,weights)+biases)
        
    #Linear Layer
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([FLAGS.hidden2,FLAGS.num_classes],stddev=1.0/math.sqrt(float(FLAGS.hidden2))),name="weights")
        biases = tf.Variable(tf.zeros([FLAGS.num_classes]),name="biases")
        logits = tf.matmul(hidden2,weights)+biases

    return logits

def lossFun(logits,labels):
    """
    Args:
      logits: predictions
      labels: ground truth labels
    """
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels,1)
    indices = tf.expand_dims(tf.range(0,batch_size),1)
    concated = tf.concat(1,[indices,labels])
    onehot_label = tf.sparse_to_dense(concated,tf.pack([batch_size,FLAGS.num_classes]),1.0,0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,onehot_label,name="cross_entropy")
    loss = tf.reduce_mean(cross_entropy)
    return loss

def train_op_Fun(loss):
    """
    Args:
      loss: loss
    Returns:
      train_op: The op for training
    """
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    global_step = tf.Variable(0,name="global_step",trainable=False)
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    """
    Args:
      logits: predictions
      labels: ground truth labels
    Returns: the correct number of true entries
    """
    correct = tf.nn.in_top_k(logits,labels,1)
    return tf.reduce_sum(tf.cast(correct,tf.int32))

def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set):
    """
    Args:
      sess: Session in which the model has been trained
      eval_correct: the number of correct predictions
      images_placeholder
      labels_placehodler
      data_set: from input_data.read_data_sets()
    Returns:
      No returns, output result directly
    """
    true_count = 0
    step_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = step_per_epoch * FLAGS.batch_size
    for step in xrange(step_per_epoch):
        images_feed,labels_feed = data_set.next_batch(FLAGS.batch_size,False)
        true_count += sess.run(eval_correct,feed_dict={images_placeholder:images_feed,labels_placeholder:labels_feed})
    precision = true_count*1.0 / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def run_train_loops():
    #read training dataset
    data_sets = input_data.read_data_sets(FLAGS.train_dir,False)

    with tf.Graph().as_default():
        images_placeholder,labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = Inference(images_placeholder)
        loss = lossFun(logits,labels_placeholder)
        train_op = train_op_Fun(loss)
        eval_correct = evaluation(logits,labels_placeholder)

        #session & Initialization
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        #train loops
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            images_feed,labels_feed = data_sets.train.next_batch(FLAGS.batch_size,False)
            _,loss_value = sess.run([train_op,loss],feed_dict={images_placeholder:images_feed,labels_placeholder:labels_feed})
            duration = time.time() - start_time

            #output loss every 100 steps
            if step%100 == 0:
                print('Traing Data Eval:')
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.train)
                print('Test Data Eval:')
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.test)

run_train_loops()
