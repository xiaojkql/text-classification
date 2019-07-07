# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-07-01 08:50:58
'''
import logging


def get_logger(log_file):
    # 再tensorflow内部的日志输出也是用的tensorflow这个logger
    logger = logging.getLogger("tensorflow")  # 将此logger的名字定义为log_name
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file,mode='a')
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # 此处的格式设置有点错误
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    handlers = [fh, sh]
    logger.handlers = handlers

    return logger
