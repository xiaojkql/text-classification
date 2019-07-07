import numpy as np
import six
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def agumentparse():
    tf.app.flags.DEFINE_string('batch_size', '4', '----')


def main(_):
    print(FLAGS.batch_size)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("infile", "123", "456")

print(("infile" in FLAGS))
# print(FLAGS.get("infile"))
print(("infil" in FLAGS))
key = "infile"


class Model:
    def __init__(self,
                 vocab_size,
                 hidden_size=1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @classmethod
    def from_dict(cls, dict_):
        config = Model(vocab_size=10)
        for (key, value) in six.iteritems(dict_):
            config.__dict__[key] = value
        return config


config_dict = {'vocab_size': 20, 'hidden_size': 50, 'other': 100}
model = Model.from_dict(FLAGS)
print(model.infile)


# tf.enable_eager_execution()

a = [[1, 2, 3], [4, 5, 6]]

b = tf.get_variable(name='a', initializer=a)

# print(b)

# print(np.random.rand(1, 2))

# print(1.0 < 2.0)
labels = [1, 2, 0]
pre_labels = [0, 2, 0]
predictions = [[4, 5, 6, 7], [10, 11, 12, 8], [45, 5, 1, 9]]
predictions = tf.cast(predictions, tf.float32)
predictions = tf.nn.softmax(logits=predictions, name='1')
log_prob = -tf.nn.log_softmax(predictions)
pre_labels = tf.math.argmax(predictions, axis=-1)
precision = tf.metrics.precision(labels, pre_labels)

with tf.Session() as sess:
    print(sess.run(predictions))
    print(sess.run(pre_labels))
    print(sess.run(log_prob))
