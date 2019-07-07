# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-07-06 18:38:48
'''
import numpy as np
import tensorflow as tf
import codecs


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


def regularizer(weights):
    """regularizer function
    Args:
        weights
    """
    loss = tf.math.reduce_sum(tf.math.square(weights))
    tf.add_to_collection(name='l2_regularization_loss', value=loss)
