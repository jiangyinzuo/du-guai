# -*- coding: utf-8 -*-
"""
加载.env环境变量
"""
import logging
import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

mode = os.environ.get('mode')
test = os.environ.get('test')

if mode == 'debug':
    logging.basicConfig(level=logging.DEBUG)
