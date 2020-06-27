# -*- coding: utf-8 -*-
"""
加载.env环境变量
"""
import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

mode = os.environ.get('mode')
test = os.environ.get('test')

cwd = os.getcwd()
parent_path = os.path.dirname(cwd)
dataset_path = parent_path + os.sep + 'dataset' + os.sep
play_dataset = dataset_path + os.environ.get('play_dataset')
follow_dataset = dataset_path + os.environ.get('follow_dataset')
