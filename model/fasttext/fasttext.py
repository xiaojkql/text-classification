# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-07-07 19:42:01
'''

import copy
import tensorflow as tf
import codecs
import numpy as np
import six
import json
from model.model_utils import embedding_lookup, regularizer


class FastTextConfig(object):

    def __init__(self,
                 vocab_size=10,
                 embedding_size=300,
                 use_pretrained_embedding='',
                 vocab_word_id_map='',
                 full_hidden_size=100,
                 dropout_rate=0.5
                 ):
        # for embedding
        # 保存embedding的文件路径， 为空字符串则不使用预训练
        self.use_pretrained_embedding = use_pretrained_embedding
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.vocab_word_id_map = vocab_word_id_map  # 提供vocab的路径，需要预先建立，整个训练语料库的词汇库

        # for full connect
        self.full_hidden_size = full_hidden_size
        self.dropout_rate = dropout_rate

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, dict_):
        config = KimCNNConfig()
        for (key, value) in six.iteritems(dict_):
            config.__dict__[key] = value
        return config


class FastText(object):
    """FastText Model
    Args:
        model_config
        is_training
        input_ids
        input_length
    """

    def __init__(self,
                 model_config,
                 is_training,
                 input_ids,
                 input_length):
        if is_training:
            model_config.dropout_prob = 0.0

        with tf.variable_scope('FastText'):
            with tf.variable_scope('embedding'):
                (self.word_embedding,
                 self.embdding_table) = embedding_lookup(input_ids=input_ids,
                                                         shape=[model_config.vocab_size,
                                                                model_config.embedding_size],
                                                         name='word-embedding')
            with tf.variable_scope('context-vector'):
                # (b,s,e) --> (b,e) / (length) -->
                output = tf.math.reduce_sum(self.word_embedding, axis=1)
                self.context_vector = output/input_length
                print(self.context_vector.shape)
            with tf.variable_scope('full-hidden-layer'):
                output = tf.layers.dense(inputs=self.context_vector,
                                         units=model_config.full_hidden_size,
                                         use_bias=True,
                                         kernel_intializer=tf.contrib.layers.xavier_initializer(),
                                         kernel_regularizer=regularizer,
                                         activation=None)
                output = tf.layers.batch_normalization(output,
                                                       training=is_training)
                output = tf.nn.dropout(tf.nn.relu(output),
                                       rate=model_config.dropout_rate)
            self.output = output

    def get_pooled_output(self):
        return self.output


def test():
    input_ids = [[1, 2, 3, 4],
                 [0, 1, 2, 3]]
    input_length = [2, 3]
    config = FastTextConfig()
    model = FastText(config, True, input_ids, input_length)


if __name__ == "__main__":
    test()
