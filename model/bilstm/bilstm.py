# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-06-28 09:59:52
'''
import copy
import tensorflow as tf
import codecs
import numpy as np
import six
import json
# import tf.contrib as contrib
# 注意模型定义与模型运行计算结果的不同


class ModelConfig(object):

    def __init__(self,
                 vocab_size=0,
                 hidden_size=300,
                 hidden_dropout_prob=0.5,
                 num_stack_layers=1,
                 use_pretrained_embedding='',
                 embedding_size=300,
                 vocab_word_id_map='',
                 num_full_layers=2,
                 full_size=100
                 ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_stack_layers
        self.hidden_dropout_size = hidden_dropout_prob

        # for embedding
        # 保存embedding的文件路径， 为空字符串则不使用预训练
        self.use_pretrained_embedding = use_pretrained_embedding
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.vocab_word_id_map = vocab_word_id_map  # 提供vocab的路径，需要预先建立，整个训练语料库的词汇库

        # for full connect
        self.full_size = full_size
        self.num_full_layers = num_full_layers

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, dict_):
        config = ModelConfig()
        for (key, value) in six.iteritems(dict_):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_FLAGS(cls, FLAGS):
        config = ModelConfig()
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


class BiLSTM(object):
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
            config.hidden_dropout_prob = 0.0

        # if config.use_pretrained_embedding ！= '':
            # 使用预训练词汇向量 --> 加载词到id映射表 --> 加载预训练词汇表 --> 建立id到词向量的映射表
        with tf.variable_scope(scope, default_name='BiLSTM'):
            with tf.variable_scope('word_embedding'):
                shape = [config.vocab_size, config.hidden_size]
                (self.word_embeddings, self.word_embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    shape=shape,
                    embedding_name='word_embedding',
                    vocab_word_id_map=config.vocab_word_id_map,
                    use_pretrained_embedding=config.use_pretrained_embedding)

            input_feats = self.word_embeddings
            with tf.variable_scope('bilstm'):
                fw_cell = create_lstm_cell(config.hidden_size, 'fw_lstm_cell')
                bw_cell = create_lstm_cell(config.hidden_size, 'bw_lstm_cell')
                _, final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell,
                    cell_bw=bw_cell,
                    dtype=tf.float32,
                    inputs=input_feats,
                    sequence_length=input_length)
                lstm_output = tf.concat(
                    axis=-1, values=[final_state[0].h, final_state[1].h])  # 提取最后一个状态的前向和后向传播的值
                print(lstm_output.shape)

            with tf.variable_scope('full-connect-1'):
                shape = [config.hidden_size*2, config.full_size]
                name = 'full-connect-1'
                full_output = full_connect_layer(
                    input=lstm_output,
                    shape=shape,
                    name=name
                )
            self.output = tf.nn.relu(full_output)

    def get_pooled_output(self):
        return self.output


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
        print("loadding pretrained vector...")
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
                if word in word_id_map:
                    # 保留给<pad>
                    embeddings[word_id_map[word]] = list(
                        map(float, word_vector))
        print("loaded")
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
    print("initialization")
    if using_xavier:
        drange = np.sqrt(np.sum(shape))
        embeddings = drange * \
            np.random.uniform(low=-1.0, high=1.0, size=shape).astype('float32')
    else:
        embeddings = np.multiply(np.add(np.random.rand(
            shape[0], shape[1]), -0.5), 0.01)  # [0,1] uniform distrib
    print("Initialized/home/qinyuan/MyProject/classification/model/kimcnn/config/model_config.json")
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
        initializer=tf.zeros_initializer)  # 当是常数的时候不需要指明shape
    output = tf.nn.xw_plus_b(input, weights, bias)
    return output
