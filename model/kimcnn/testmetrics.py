import tensorflow as tf


label = [1, 2, 3, 4, 5, 6]
pred = [0, 2, 0, 0, 0, 0]
pre= tf.metrics.precision(label, pred)

print(pre)
