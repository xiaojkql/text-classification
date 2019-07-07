# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-07-01 08:48:49
'''

import tensorflow as tf
import numpy as np
from utils import get_logger
from kimcnn import ModelConfig, KimCNN, full_connect_layer
import collections
import re
import os
import codecs
import pandas as pd

logger_file = 'log/main.log'
if not os.path.exists(os.path.dirname(logger_file)):
    os.mkdir(os.path.dirname(logger_file))
logger = get_logger(logger_file)
FLAGS = tf.app.flags.FLAGS  # 获得标志FLAGS
# tf.app.flags.DEFINE_string()
# tf.app.run()  --> main(_)


def create_model(model_config, is_training, input_ids, input_length, labels, num_labels):
    # 创建模型计算输出结果
    model = KimCNN(config=model_config, is_training=is_training,
                   input_ids=input_ids, input_length=input_length)
    output = model.get_pooled_output()

    with tf.variable_scope("loss"):
        if is_training:
            pass
            # 加一个dropout
            # 需要提供一个参数，dropoutpro
            # output = tf.nn.dropout(output, keep_prob=0.9)
        shape = [output.shape.as_list()[-1], num_labels]
        name = "logits-layer"
        logits = full_connect_layer(output, shape, name)
        probabilities = tf.nn.softmax(logits)
        log_probs = -tf.nn.log_softmax(logits)
        if labels is not None:
            one_hot_labels = tf.one_hot(
                labels, depth=num_labels, dtype=tf.float32)
            # per_example_loss = tf.math.reduce_mean(
            #     one_hot_labels*log_probs, axis=-1)
            # total_loss = tf.math.reduce_mean(per_example_loss)
            total_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=one_hot_labels, logits=logits)
        else:
            total_loss = None
    return (total_loss, logits, probabilities)


# 模型初始化部分
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()

    for var in tvars:  # tf.trainable_variables() 返回的变量
        name = var.name
        # pattern, string， 保存的checkpoint的变量名的格式
        m = re.match("^(.*):\\d+$", name)
        if m is not None:  # 匹配到了
            name = m.group(1)
        name_to_variable[m] = var
    init_vars = tf.train.list_variables(
        init_checkpoint)  # return tuple of (name, shape)
    assignment_map = collections.OrderedDict()

    for x in init_vars:
        name, shape = x[0], x[1]
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name+':0'] = 1

    return (assignment_map, initialized_variable_names)


# 建立模型优化器部分
def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps=10):
    """建立模型的优化器
    Args
        loss: 损失函数
        init_lr: 初始学习率
        num_train_steps: 总共的优化步长
        num_warmup_steps: 预热步长， if global_step < num_warmup_steps: lr = (global_step/nums_warmup_steps) * lr
    """
    # 全局步长变量 --> 学习率(常数) --> 学习率变化方式 --> 判断预热学习率
    # --> 定义优化器类 --> 计算梯度
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate=learning_rate,
        global_step=global_step,
        decay_steps=num_train_steps,  # 衰减步长，即训练的总步数
        end_learning_rate=0.0,
        power=1.0,
        cycle=False
    )
    if num_warmup_steps:
        global_step_float = tf.cast(global_step, tf.float32)  # 类型转换
        num_warmup_steps_float = tf.constant(
            value=num_warmup_steps, dtype=tf.float32)  # 赋值一个变量
        warmup_percent = global_step_float / num_warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent
        is_warmup = tf.cast(
            (global_step_float < num_warmup_steps_float), tf.float32)
        learning_rate = (1.0 - is_warmup) * learning_rate + \
            is_warmup * warmup_learning_rate

    # 需要对不同的变量使用不同的学习率
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    embed_vars = [var for var in tf.trainable_variables()
                  if 'embedding' in var.name]
    other_vars = [var for var in tf.trainable_variables()
                  if 'embedding' not in var.name]
    tvars = tf.trainable_variables()  # a list of
    gradients = tf.gradients(loss, tvars)  # a list of gradient
    grads, global_norm = tf.clip_by_global_norm(gradients, clip_norm=1)  #
    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)  # 返回的是operation, 当梯度下降优化完成后，会自动对global_step加1  # 和miniza差不多， minimize 是直接用在损失函数，而此先计算各个变量的梯度后再进行梯度下降
    return train_op


def model_fn_builder(model_config, num_labels, init_checkpoint, init_lr, num_train_steps, num_warmup_steps):
    def model_fn(features, labels, mode, params):
        """ 标准的四个参数，features,mode,params """
        # feature: input_ids, input_length
        logger.info("-"*8 + "features" + "-"*8)
        # features: key2tensor
        # for name in features:
        # logger.info(" name = %s, shape = %s" %
        # (name, features[name].shape))
        input_ids = features[1]
        print(input_ids.shape)
        input_length = tf.squeeze(features[0], axis=-1)
        if tf.estimator.ModeKeys.TRAIN == mode or tf.estimator.ModeKeys.EVAL == mode:
            labels_ids = tf.squeeze(labels, axis=-1) - 1  # label从
        else:
            labels_ids = None

        """ 定义模型，计算损失，概率， """
        is_training = (tf.estimator.ModeKeys.TRAIN == mode)
        (total_loss, logits, probabilities) = create_model(
            model_config,
            is_training,
            input_ids,
            input_length,
            labels_ids,
            num_labels)  # moxingjisuanbufende输出是一致的，但是接下来就要选择了
        pre_labels = tf.math.argmax(logits, axis=-1)

        """ 用初始化文件初始化模型参数 """
        # 上面将模型定义完毕后，才能有模型中的变量
        tvars = tf.trainable_variables()  # 可训练的模型参数
        initialized_variable_names = {}
        if init_checkpoint:  # 当有预训练的时候从此处进行先初始化
            logger.info("init checkpoint")
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            # 需要提供两个参数一是初始化文件路径，二是初始化map
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            logger.info("init done")

        """ 输出模型参数, 计数模型中总的参数个数 """
        total_parameters = 0
        for var in tvars:
            variable_parameters = 1
            for dim in var.get_shape().as_list():
                variable_parameters *= dim
            total_parameters += variable_parameters
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT"
            logger.info(" name = %s, shape = %s%s" %
                        (var.name, var.shape, init_string))
        logger.info("total parameters %d" % (total_parameters))
        output_spec = None

        metrics = {"eval_precision": tf.metrics.precision(labels_ids, predictions=pre_labels),
                   "accuracy": tf.metrics.accuracy(labels_ids, pre_labels),
                   'recall': tf.metrics.recall(labels_ids, pre_labels)}

        """ for train: train_op --> train-hooks -->  EstimatorSpec"""
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = create_optimizer(
                total_loss, init_lr, num_train_steps, num_warmup_steps)
            """ 日志hook """
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()

            """ summary-hook """
            # 定义一个summary --> 用merge_all进行收集 --> 设置定义summary-hook
            tf.summary.scalar('loss', total_loss)
            for metrix_name, op in metrics.items():
                tf.summary.scalar(metrix_name, op[1])
                hook_dict[metrix_name] = op[1]

            summary_op = tf.summary.merge_all()

            hook_dict['labels_id'] = labels_ids
            hook_dict['pre_labels'] = pre_labels
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=1)  # 需要设置打印步长数 同样是tf.train.SessionRunHook 可以在Estimatorspec中定义，也可以再estimator中定义

            summary_hook = tf.train.SummarySaverHook(
                save_steps=1, output_dir=os.path.join(FLAGS.output_dir, 'summary'), summary_op=summary_op)
            output_spec = tf.estimator.EstimatorSpec(
                mode, loss=total_loss, train_op=train_op, training_hooks=[
                    logging_hook, summary_hook], eval_metric_ops=metrics
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            hook_dict = {}
            hook_dict['loss'] = total_loss
            # hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            hook_dict['labels_id'] = labels_ids
            hook_dict['pre_labels'] = pre_labels
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=1)

            def metric_fn(label_ids, pre_labels):

                return {"eval_precision": tf.metrics.precision(label_ids, predictions=pre_labels),
                        "accuracy": tf.metrics.accuracy(label_ids, pre_labels)}
            output_spec = tf.estimator.EstimatorSpec(
                mode, loss=total_loss, eval_metric_ops=metric_fn(labels_ids, pre_labels), training_hooks=[logging_hook])
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode, predictions=pre_labels)
        return output_spec
    return model_fn


# 清理保存的模型内容
def clean():
    if FLAGS.do_clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            def del_file(path):
                file_ls = os.listdir(path)
                for file_ in file_ls:
                    c_path = os.path.join(path, file_)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(FLAGS.output_dir)
            except Exception as identifier:
                exit(-1)


def file_based_input_fn_builder(input_file, is_training, drop_remainder=False):
    features = {
        "labels": tf.FixedLenSequenceFeature([], tf.int64),
        "sent_len": tf.FixedLenSequenceFeature([], tf.int64),
        "word_ids": tf.FixedLenSequenceFeature([], tf.int64),
        "char_ids": tf.FixedLenSequenceFeature([], tf.int64)
    }

    def _parse(ex):
        _, example = tf.parse_single_sequence_example(serialized=ex,
                                                      sequence_features=features)
        for name in example.keys():
            if example[name].dtype == tf.int64:
                example[name] = tf.to_int32(example[name])
        return (example['sent_len'], example['word_ids'], example['char_ids']), example['labels']

    def input_fn(params):
        shape = (([None], [None], [None]), [None])
        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.repeat(
                params['train_epochs']).shuffle(buffer_size=1000)
        dataset = dataset.map(_parse).padded_batch(
            params['batch_size'], shape, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    return input_fn


def train():
    """
     清理空间 --> 建立输出文件夹 --> 设置session config(运行设置，线程这些参数)
     --> 设置estimator的Config (模型保存这些设置) --> 定义模型 --> 定义estimator
     --> 分两条路进行： train + eavl   and   predict
     for train+eval: train_input_fn
    """
    num_train_examples = 0
    with codecs.open(FLAGS.size_file_path, 'r', 'utf-8') as in_f:
        num_train_examples = int(in_f.readline()[:-1])

    if FLAGS.do_train and FLAGS.do_clean:
        clean()
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    # 设置session的run
    session_config = tf.ConfigProto(
        # log_device_palcement=True,
        # inter_op_parrallelism_threads=0,
        allow_soft_placement=True
    )

    # 设置run Config
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(FLAGS.output_dir, "model"),
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=session_config,
        log_step_count_steps=10
    )

    # 定义模型 model_fn
    if FLAGS.load_params_from_json:
        model_config = ModelConfig.from_json_file(FLAGS.json_file)
    num_train_steps = int(num_train_examples *
                          FLAGS.train_epochs / FLAGS.batch_size)
    num_warmup_steps = num_train_steps * FLAGS.num_warmup_proportion
    model_fn = model_fn_builder(
        model_config=model_config,
        num_labels=19,
        init_checkpoint=FLAGS.init_checkpoint,
        init_lr=FLAGS.init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps
    )

    # 定义estimator
    params = dict((('batch_size', FLAGS.batch_size),
                   ('train_epochs', FLAGS.train_epochs)))
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config
    )

    # for train and eval
    if FLAGS.do_train and FLAGS.do_eval:
        train_input_fn = file_based_input_fn_builder(
            FLAGS.train_file_path, is_training=True)
        eval_input_fn = file_based_input_fn_builder(
            FLAGS.eval_file_path, is_training=False)

        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',  # 可以自己定义metric_fn，定义一个dict
            max_steps_without_decrease=5,
            run_every_secs=None,
            run_every_steps=1000  # 设置运行evaluation的步长
        )

        trainSpec = tf.estimator.TrainSpec(
            train_input_fn, hooks=[early_stopping_hook])
        evalSpec = tf.estimator.EvalSpec(eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, trainSpec, evalSpec)
    elif FLAGS.do_eval:
        eval_input_fn = file_based_input_fn_builder(
            FLAGS.eval_file_path, is_training=False)
        estimator.evaluate(eval_input_fn)
    elif FLAGS.do_predict:
        pre_input_fn = file_based_input_fn_builder(
            FLAGS.eval_file_path, is_training=False)
        pre_labels = list(estimator.predict(pre_input_fn))
        result = pd.DataFrame(data=pre_labels, columns=['class'])
        result.to_csv('./submmit/result.csv')


def argument_parse():
    tf.app.flags.DEFINE_boolean(
        'do_train', True, 'Whether to train the model?')
    tf.app.flags.DEFINE_boolean(
        'do_eval', True, 'Whether to evaluate the model when training?')
    tf.app.flags.DEFINE_boolean(
        'do_predict', True, 'Whether to do prediction with the model?')
    tf.app.flags.DEFINE_boolean(
        'do_clean', True, 'Whether to clean the output?')

    tf.app.flags.DEFINE_string(
        'output_dir', './output', 'The dir to store the result.')
    tf.app.flags.DEFINE_integer(
        'save_summary_steps', 1, 'The steps to save summary of the model')
    tf.app.flags.DEFINE_integer(
        'save_checkpoints_steps', 500, 'The steps to save checkpoint of the model')

    tf.app.flags.DEFINE_string(
        'init_checkpoint', '', 'The path of pretrained weights of the model.')
    tf.app.flags.DEFINE_float(
        'init_lr', 0.01, 'The steps to save checkpoint of the model')
    tf.app.flags.DEFINE_integer(
        'train_epochs', 10, 'The epochs of training the model')
    tf.app.flags.DEFINE_integer(
        'batch_size', 128, 'Batch size')
    tf.app.flags.DEFINE_float(
        'num_warmup_proportion', 1, 'The warmup steps')

    tf.app.flags.DEFINE_string(
        'train_file_path', '/home/qinyuan/MyProject/classification/dataset/train/example.tfrecords', 'The path of train dataset.')
    tf.app.flags.DEFINE_string(
        'test_file_path', '/home/qinyuan/MyProject/classification/dataset/test/example.tfrecords', 'The path of test dataset.')
    tf.app.flags.DEFINE_string(
        'eval_file_path', '/home/qinyuan/MyProject/classification/dataset/dev/example.tfrecords', 'The path of dev dataset.')
    tf.app.flags.DEFINE_string(
        'size_file_path', '/home/qinyuan/MyProject/classification/dataset/train/size.txt', 'The path of size file.')

    tf.app.flags.DEFINE_boolean(
        'drop_remainder', False, 'Whether to drop the remaining of the dataset?')

    # for model setting
    tf.app.flags.DEFINE_boolean(
        'load_params_from_json', True, 'Whether load model params from json file')
    tf.app.flags.DEFINE_string(
        'json_file', '/home/qinyuan/MyProject/classification/model/kimcnn/config/model_config.json', 'The path of params json file.')
    tf.app.flags.DEFINE_integer(
        'vocab_size', 0, 'The epochs of training the model')  # 必须提供的
    tf.app.flags.DEFINE_integer(
        'hidden_size', 300, 'The epochs of training the model')
    tf.app.flags.DEFINE_float(
        'hidden_dropout_prob', 0.5, 'The epochs of training the model')
    tf.app.flags.DEFINE_integer(
        'num_stack_layers', 1, 'The epochs of training the model')
    tf.app.flags.DEFINE_integer(
        'embedding_size', 300, 'The epochs of training the model')
    tf.app.flags.DEFINE_integer(
        'full_size', 1, 'The epochs of training the model')
    tf.app.flags.DEFINE_integer(
        'num_full_layers', 1, 'The epochs of training the model')
    tf.app.flags.DEFINE_string(
        'use_pretrained_embedding', '', 'The path of dev dataset.')
    tf.app.flags.DEFINE_string(
        'vocab_word_id_map', '', 'The path of dev dataset.')


def main(_):
    train()


if __name__ == "__main__":
    argument_parse()
    tf.app.run()
