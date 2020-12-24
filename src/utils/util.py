import logging
import os
import time
from datetime import datetime

from src.config.config import ROOT_PATH


def format_time(time_str: str):
    if type(time_str) != 'str':
        return time_str
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return datetime.strptime(time_str, '%Y-%m-%d')


def get_current_time() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))


def create_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


def save_dict_to_txt(path: str, data: dict):
    f = open(path + '.txt', 'w')
    f.write(str(data))
    f.close()


def read_txt_to_dict(path: str) -> dict:
    f = open(path + '.txt', 'r')
    dict_ = eval(f.read())
    f.close()
    return dict_


def save_config(key: str, value: object):
    config = read_txt_to_dict(ROOT_PATH + '/src/config/env')
    logging.info('key: {}, value: {}, config: {}'.format(key, value, config))
    if value:
        config[key] = value
    else:
        del config[key]

    logging.info('key: {}, value: {}, config: {}'.format(key, value, config))
    save_dict_to_txt(ROOT_PATH + '/src/config/env', config)


def read_config(key: str) -> str:
    """
    在配置文件里面读取自定义的配置
    :param key: 配置的名称
    :return:
    """
    config = read_txt_to_dict(ROOT_PATH + '/src/config/env')
    return config.get(key, None)


def read_mapping():
    mapping = {}
    # 在Windows机器上就会出错，所以这里需要指定下encoding
    with open(ROOT_PATH + '/src/config/mapping.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            mapping[line.split(',')[0]] = line.split(',')[1]
    return mapping


def name_list_2_plc_list(name_list):
    return [NAME_TO_PLC[name] for name in name_list]


# 读取Mapping关系
PLC_TO_NAME = read_mapping()
NAME_TO_PLC = {v: k for k, v in PLC_TO_NAME.items()}
