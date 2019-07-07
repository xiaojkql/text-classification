# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-07-02 20:17:20
'''


import copy
import tensorflow as tf
import codecs
import numpy as np
import six
import json
# import tf.contrib as contrib
# 注意模型定义与模型运行计算结果的不同


class KimCNNConfig(object):

    def __init__(self,
                 vocab_size=10,
                 embedding_size=300,
                 use_pretrained_embedding='',
                 vocab_word_id_map='',
                 filters_size=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 full_layers_size=[100],
                 full_dropout_prob=0.5
                 ):
        # for embedding
        # 保存embedding的文件路径， 为空字符串则不使用预训练
        self.use_pretrained_embedding = use_pretrained_embedding
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.vocab_word_id_map = vocab_word_id_map  # 提供vocab的路径，需要预先建立，整个训练语料库的词汇库

        # for filter
        self.filters_size = filters_size
        self.num_filters = num_filters

        # for full connect
        self.full_layers_size = full_layers_size
        self.full_dropout_prob = full_dropout_prob

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

    @classmethod
    def from_FLAGS(cls, FLAGS):
        config = KimCNNConfig()
        config.hidden_size = FLAGS.hidden_size
        config.num_hidden_layers = FLAGS.num_stack_layers
        config.hidden_dropout_size = FLAGS.hidden_dropout_prob

        # for embedding
        # 保存embedding的文件路径， 为空字符串则不使用预训练
        config.use_pretrained_embedding = FLAGS.use_pretrained_embedding
        config.vocab_size = FLAGS.vocab_size
        config.embedding_size = FLAGS.embedding_size
        config.vocab_word_id_map = FLAGS.vocab_word_id_map  # 提供vocab的路径，需要预先建立，整个训练语料库的词汇库

        # for full connect
        config.full_size = FLAGS.full_size
        config.num_full_layers = FLAGS.num_full_layers
        return config


""" 采用estimator来写模型时，只要在模型定义时，写好features, labels, mode, params这些东西的具体流向就好了 """


class KimCNN(object):
    """ 采用Bert的模式来构建代码 """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_length,
                 scope=None):
        """Constructor for bilstm model
        Args:
            config:
            is_training
        """
        if not is_training:  # 如果是training则设置dropout为0
            config.full_dropout_prob = 0.0

        # if config.use_pretrained_embedding ！= '':
            # 使用预训练词汇向量 --> 加载词到id映射表 --> 加载预训练词汇表 --> 建立id到词向量的映射表
        with tf.variable_scope(scope, default_name='KimCNN'):
            with tf.variable_scope('embedding'):
                shape = [config.vocab_size, config.embedding_size]
                (self.word_embeddings, self.word_embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    shape=shape,
                    embedding_name='word_embedding',
                    vocab_word_id_map=config.vocab_word_id_map,
                    use_pretrained_embedding=config.use_pretrained_embedding)

            input_feats = self.word_embeddings
            with tf.variable_scope('CNN-layer'):
                cnn_output = []
                for filter_size, num_filter in zip(config.filters_size, config.num_filters):
                    output = tf.layers.conv1d(inputs=input_feats,
                                              filters=num_filter,
                                              kernel_size=filter_size,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              strides=1,
                                              padding='valid',
                                              activation='relu'
                                              )
                    pool = tf.keras.layers.GlobalMaxPool1D()  # 自动降了一维 这里用了max_pooling 也可以用top_k来挑选
                    output = pool(output)
                    cnn_output.append(output)
                self.cnn_output = tf.concat(cnn_output, axis=-1)
            input_feats = self.cnn_output
            # cnn 层后是full-connect
            # linear --> batch_normalization
            for idx, full_size in enumerate(config.full_layers_size):
                with tf.variable_scope("full-connect-{}".format(idx)):
                    input_feats = tf.layers.dense(inputs=input_feats,
                                                  units=full_size,
                                                  use_bias=True,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  activation=None)
                    input_feats = tf.nn.tanh(tf.layers.batch_normalization(
                        input_feats, training=is_training))
                    input_feats = tf.layers.dropout(inputs=input_feats,
                                                    rate=config.full_dropout_prob)
            self.output = input_feats

    def get_pooled_output(self):
        return self.output

    def get_cnn_output(self):
        return self.cnn_output


def create_lstm_cell(hidden_size, name, reuse=False):
    return tf.nn.rnn_cell.BasicLSTMCell(hidden_size, name=name, reuse=reuse)


def embedding_lookup(input_ids,
                     shape,
                     embedding_name,
                     vocab_word_id_map,
                     use_pretrained_embedding=''):
    embeddings = embedding_value(shape, using_xavier=True)  # 整个初始化的值
    embeddings[0] = 0
    if use_pretrained_embedding != '':
        word_id_map = {}
        with codecs.open(vocab_word_id_map, 'r', encoding='utf-8') as in_f:
            for index, word in enumerate(in_f):
                word_id_map[word.strip()] = index
        with codecs.open(use_pretrained_embedding, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                line = line.strip().split(" ")
                if not line:
                    continue
                word, word_vector = line[0], line[1:]
                if len(word_vector) < 10:
                    continue
                if word in word_id_map:
                    # 保留给<pad>
                    embeddings[word_id_map[word]] = list(
                        map(float, word_vector))
    # zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, shape[1]])
    # random_unk = tf.get_variable(name='unk_emb', shape=[
    #  1, shape[1]], initializer=tf.truncated_normal_initializer())

    embedding_table = tf.get_variable(
        name=embedding_name, initializer=embeddings)
    # embedding_table = tf.concat(0, [zero_pad, word_embeddings])
    output_embeddings = tf.nn.embedding_lookup(embedding_table, input_ids)
    return (output_embeddings, embedding_table)
    # 初始化词向量


def embedding_value(shape, using_xavier=True):
    if using_xavier:
        drange = np.sqrt(np.sum(shape))
        embeddings = drange * \
            np.random.uniform(low=-1.0, high=1.0, size=shape).astype('float32')
    else:
        embeddings = np.multiply(np.add(np.random.rand(
            shape[0], shape[1]), -0.5), 0.01)  # [0,1] uniform distrib
    return embeddings


def initialize_weights(shape, name, init_type='xavier'):
    """weights initializer
    Args:
        shape:
        name:
        init_type:[random, xavier,varscale]
    """
    name = name + '_weights'
    if init_type == 'random':
        weights = tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    elif init_type == 'xavier':
        weights = tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer()
        )
    elif init_type == 'varscale':
        weights = tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer()
        )

    return weights


def full_connect_layer(input, shape, name):
    weights = initialize_weights(shape, name)
    bias = tf.get_variable(
        name=(name+'_bias'),
        shape=[shape[1]],
        initializer=tf.zeros_initializer())  # 当是常数的时候不需要指明shape
    output = tf.nn.xw_plus_b(input, weights, bias)
    return output
