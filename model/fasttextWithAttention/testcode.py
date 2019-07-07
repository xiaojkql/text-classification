import numpy as np
import tensorflow as tf

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = np.array([[1], [2], [3]])

# print(a*b)

# a = np.array([[[1], [2], [3]], [[7], [8], [9]], [[10], [11], [12]]])
# b = np.array([[[1, 4]], [[2, 8]], [[10, 11]]])
# print(a*b)

# a = np.array([[1, 2, 3], [4, 5, 6]])
# a = np.transpose(a, axes=[1, 0])
# print(a)

a = tf.get_variable(name='a', initializer=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = tf.expand_dims(a, axis=0)
c = tf.get_variable(name='a', initializer=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(b)
# print(a)

# with tf.Session() as sess:
#     print(sess.run(a))


# # a = tf.constant(10)
# init_op = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init_op)
# print(sess.run(a))
# print(sess.run(b))
