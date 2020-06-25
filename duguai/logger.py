# -*- coding: utf-8 -*-
import logging
import traceback


def log_locals(err: Exception):
    """
    调试时输出本地变量的值
    @param err: 异常
    """
    flag = True
    for tb in traceback.walk_tb(err.__traceback__):
        tb_frame = tb[0]
        logging.error('co_name: {}; f_lineno: {}'.format(tb_frame.f_code.co_name, tb_frame.f_lineno))
        f_locals: dict = tb_frame.f_locals

        # 忽略第一个栈
        if flag:
            flag = False
            continue
        for k, v in f_locals.items():
            if hasattr(v, '__dict__'):
                logging.error('{}: {}'.format(k, v.__dict__))
            else:
                logging.error('{}: {}'.format(k, v))
