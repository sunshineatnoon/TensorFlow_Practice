import tensorflow as tf
import numpy as np

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
x = tf.Variable(tf.random_normal([3,3]))
sess = tf.Session()
sess.run(tf.initialize_all_variables())


matrix = np.random.rand(3,3)
for i in range(3):
    for j in range(3):
        print matrix[i][j]

x = tf.assign(x,matrix)
x = tf.reshape(x,[-1,3,3,1])
max_pool = tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME')
return_value = sess.run(max_pool)  # or `assign_op.op.run()`
print return_value
