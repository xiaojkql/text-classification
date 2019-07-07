# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-07-06 17:08:09
'''
import json
import six
import tensorflow as tf
from model.model_utils import embedding_lookup, initialize_weights, regularizer

# 模型这一部分只负责计算到logits为止，其余的都在外面进行计算


class FastTextAttenConfig(object):

    def __init__(self,
                 vocab_size=None,
                 embedding_size=300,
                 use_pretrained_embedding='',
                 vocab_word_id_map='',
                 attention_sizes=[50],
                 atten_hidden_sizes=[50],
                 num_attention_vectors=5,
                 full_layers_size=[100],
                 dropout_prob=0.5  # 只需要用一个
                 ):
        # for embedding
        # 保存embedding的文件路径， 为空字符串则不使用预训练
        self.use_pretrained_embedding = use_pretrained_embedding
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.vocab_word_id_map = vocab_word_id_map  # 提供vocab的路径，需要预先建立，整个训练语料库的词汇库

        # for attention layers
        self.attention_sizes = attention_sizes
        self.atten_hidden_sizes = atten_hidden_sizes
        self.num_attention_vectors = num_attention_vectors

        # for full connect
        self.full_layers_size = full_layers_size
        self.dropout_prob = dropout_prob

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, dict_):
        config = FastTextAttenConfig()
        for (key, value) in six.iteritems(dict_):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_FLAGS(cls, FLAGS):
        config = FastTextAttenConfig()
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


class FastTextWithAttention(object):
    """Model Class
    Args:
        model_config
        word_ids
        word_length
     """

    def __init__(self,
                 model_config,
                 word_ids,
                 word_length,
                 is_training):
        """Model Class
        Args:
            model_config
            word_ids
            word_length
        """
        if is_training:
            model_config.dropout_prob = 0
        with tf.variable_scope("model-FastTextAtten"):
            with tf.variable_scope("embedding-layer"):
                shape = [model_config.vocab_size, model_config.embedding_size]
                embedding_name = "word-embedding"
                self.word_embedding, self.lookup_table = embedding_lookup(input_ids=word_ids,
                                                                          shape=shape,
                                                                          embedding_name=embedding_name,
                                                                          vocab_word_id_map=model_config.vocab_word_id_map,
                                                                          use_pretrained_embedding=model_config.use_pretrained_embedding)
            with tf.variable_scope("attention-layer"):
                atten_weights = self.word_embedding
                for idx, attention_size in enumerate(model_config.attention_sizes):
                    atten_weights = tf.layers.dense(inputs=atten_weights,
                                                    units=attention_size,
                                                    activation='tanh',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='attention-weights-linear-{}'.format(
                                                        idx),
                                                    kernel_regularizer=regularizer)
                    atten_weights = tf.nn.dropout(x=atten_weights,
                                                  rate=model_config.dropout_prob,
                                                  name='dropout-after-attention-weights-linear-{}'.format(idx))  # name for this operation
                atten_weights = tf.layers.dense(inputs=atten_weights,
                                                units=model_config.num_attention_vectors,
                                                activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                kernel_regularizer=regularizer,
                                                name='attention-weights-num-attention-vectors')
                atten_weights = tf.nn.softmax(atten_weights, axis=1)
                # (batch_size, seq_len, num_vectors)
                # (batch_size, seq_len, embedding_size)
                # --> (batch_size, num_vectors, embedding_size)
                # (b,n,s) ** (b,s,e)
                context_vectors = tf.matmul(tf.transpose(atten_weights, [0, 2, 1]),
                                            self.word_embedding)
                # (b, n, e) --> (b,n,h)
                for idx, hidden_size in enumerate(model_config.atten_hidden_sizes):
                    context_vectors = tf.layers.dense(inputs=context_vectors,
                                                      units=hidden_size,
                                                      activation=None,
                                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                      kernel_regularizer=regularizer,
                                                      name='atten-hidden_layer-{}'.format(idx))
                    context_vectors = tf.layers.batch_normalization(inputs=context_vectors,
                                                                    training=is_training)
                    context_vectors = tf.nn.dropout(tf.nn.tanh(context_vectors),
                                                    rate=model_config.dropout_prob)
                # (b,n,h) --> (b,n*h)
                shape = [-1, model_config.num_attention_vectors *
                         model_config.atten_hidden_sizes[-1]]
                self.context_vectors = tf.reshape(context_vectors, shape)

            with tf.variable_scope('full-connect-layer'):
                full_output = self.context_vectors
                for idx, size in enumerate(model_config.full_layers_size):
                    full_output = tf.layers.dense(inputs=full_output,
                                                  units=size,
                                                  activation='tanh',
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  kernel_regularizer=regularizer)
                    full_output = tf.nn.dropout(full_output,
                                                rate=model_config.dropout_prob)
            self.output = full_output

    def get_pooled_output(self):
        return self.output
