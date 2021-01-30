from typing import Tuple

import pandas as pd
import numpy as np
import logging
from scipy.stats import skew
from scipy.stats import kurtosis
from datetime import datetime

from sklearn.model_selection import train_test_split

from src.utils.util import name_list_2_plc_list, format_time

feature_name_columns = ['最终烟丝含水实际值', '烘丝出口温度', '瞬时流量', '累计流量',
                        '罩压力反馈值', '罩压力实际值', '工艺气体温度反馈值', '工艺气体温度实际值',
                        '入口烟丝水分', '回潮机水分增加', '罩压力PID控制反馈值', '工艺气速度反馈值',
                        '工艺气速度实际值', '工艺气速度设定值', '工艺气体温度设定值', '最终烟丝含水设定值',
                        '桶温区1阀后蒸汽压力', '桶温区2阀后蒸汽压力', '工作点脱水', '脱水速度',
                        '工艺蒸汽流量实际值', '工艺蒸汽流量设定值', 'SIROX阀后蒸汽压力', 'Sirox出口物料温度']
label_column = ['桶温区1设定值', '桶温区2设定值']

BRAND = '牌号'
BATCH = '批次号'
HUMIDITY = '最终烟丝含水实际值'
TIME = 'timestamp'
WORD_STATUS = '生产模式1'
WORD_STATUS_PRODUCE = 32

feature_plc_columns = name_list_2_plc_list(feature_name_columns)

#############################
## 模型所需的超参数
STABLE_WINDOWS_SIZE = 10  # 稳态的时长
SPLIT_NUM = 10  # 特征选取分割区间的数量（需要被FEATURE_RANGE整除）
TIME_IN_ROLLER = 70  # 烟丝在一个滚筒的时间
MODEL_CRITERION = 0.05  # 模型标准，工艺标准为0.5
FEATURE_RANGE = 60  # 特征选取的区间范围
LABEL_RANGE = 10  # Label选取的区间范围
SETTING_LAG = 20  # 设定值和实际值的时延
REACTION_LAG = 10  # 实际值调整后，水分变化的时延
FURTHER_STEP = 10  # 未来时刻出口水分采样步长

MODEL_TRANSITION_CRITERION = 0.05
TRANSITION_FEATURE_RANGE = 16  # Transition 特征选取的区间范围
TRANSITION_SPLIT_NUM = 4  # Transition 特征选取分割区间的数量
STABLE_UNAVAILABLE = 200  # 出口水分不可用阶段
TRANSITION_SIZE = 400  # 定义 Transition 的长度

MODEL_HEAD_CRITERION = 0.25
#############################

def calc_feature_lgbm(item_: pd.DataFrame) -> np.array:
    """
    测试阶段用于计算特征的
    calc feature for each sample data
    :param item_: sample data
    :return: feature array
    """
    return item_[feature_plc_columns].values.ravel()


def calc_feature_lr(item_: pd.DataFrame, split_num: int, start: int = None, end: int = None) -> np.array:
    """
    测试阶段用于计算特征的
    calc feature for each sample data
    :param item_: sample data
    :param start
    :param end
    :param split_num: how many splits after splitting
    :return: feature array
    """
    if start and end:
        feature_slice = item_[feature_plc_columns].iloc[start: end].values
    else:
        feature_slice = item_[feature_plc_columns].values

    # shape = (SPLIT_NUM, FEATURE_RANGE / SPLIT_NUM, FEATURE_NUM)
    feature_slice = np.array(np.vsplit(feature_slice, split_num))

    # shape = (5, SPLIT_NUM, FEATURE_NUM)
    # 比如，feature前80个都是均值，在这80个里面，被分为了SPLIT_NUM段，每一段都是FEATURE_NUM个features
    feature = np.concatenate([
        np.mean(feature_slice, axis=1).ravel(),
        np.std(feature_slice, axis=1).ravel(),
        calc_integral(feature_slice).ravel(),
        skew(feature_slice, axis=1).ravel(),
        kurtosis(feature_slice, axis=1).ravel(),
    ])

    return feature.ravel()


def calc_integral(data_: np.array) -> np.array:
    """
    calc integral
    :param data_: shape = (SPLIT_NUM, FEATURE_RANGE / SPLIT_NUM, FEATURE_NUM)
    :return shape = (SPLIT_NUM, FEATURE_NUM), each value is the integral
    """
    if data_.shape[0] <= 1:
        return 0
    sum_ = np.sum(data_, axis=1)
    return sum_ - (data_[:, 0, :] + data_[:, data_.shape[1] - 1, :]) / 2


def concatenate(data_: list) -> np.array:
    """
    concatenate list with item of different length
    """
    result = data_[0]
    for i in range(1, len(data_)):
        if len(data_[i]) is not 0:
            result = np.concatenate([result, data_[i]], axis=0)
    return result


def clip_last(pred: np.array, last_temp_1: float, last_temp_2: float) -> np.array:
    if len(pred) is not 2:
        raise Exception('Predicted value MUST have 2 value')
    pred[0] = pred[0] * 0.7 + last_temp_1 * 0.3
    pred[1] = pred[1] * 0.7 + last_temp_2 * 0.3
    return pred


def clip(pred: np.array, criterion_1: float, criterion_2: float, bound: float = 2.0) -> np.array:
    """
    clip the predicted to avoid over-estimated
    :param bound:
    :param pred: predicted values
    :param criterion_1: 一区温度标准
    :param criterion_2: 二区温度标准
    :return: clipped values
    """
    if len(pred) is not 2:
        raise Exception('Predicted value MUST have 2 value')
    pred[0] = np.clip(pred[0], criterion_1 - bound, criterion_1 + bound)
    pred[1] = np.clip(pred[1], criterion_2 - bound, criterion_2 + bound)
    return pred


def adjust(pred: list, original_humid: list, setting: float) -> list:
    """
    简单粗暴加个惩罚项，出口水分连续5个点超过了某个阈值，就加个惩罚项纠正下。
    :param pred:
    :param setting: 出口水分设定值
    :param original_humid: 出口水分原始值
    :return:
    """
    if len(original_humid) == 0:
        return pred
    # if len(original_humid) != FEATURE_RANGE:
    #    return pred
    original_humid = original_humid[-5:]
    if np.all(original_humid == 0):
        return pred
    original_humid_diff = np.array([i - setting for i in original_humid])
    ratio = 0
    # ratio = 1.2
    if np.all(original_humid_diff > 0.05):
        pred[0] += np.sum(original_humid_diff) * ratio
        pred[1] += np.sum(original_humid_diff) * ratio
    if np.all(original_humid_diff < 0.05):
        pred[0] -= np.sum(original_humid_diff) * ratio
        pred[1] -= np.sum(original_humid_diff) * ratio
    return pred


# ----------------训练相关函数------------------

def split_data_by_brand(data: pd.DataFrame) -> dict:
    """
    split the continuous time series data into each brand and batch (牌号和批次)
    :param data: the continuous time series data
    :return: data after split, data_per_brand[i][j] means i-th brand and j-th batch
    """

    data_per_brand = {}

    for brand in data[BRAND].unique():
        item_brand = data[data[BRAND] == brand]
        data_per_batch = []

        for batch in item_brand[BATCH].unique():
            item_batch = item_brand[item_brand[BATCH] == batch]

            item_batch[TIME] = item_batch[TIME].map(lambda x: format_time(x))
            item_batch = item_batch.sort_values(by=[TIME], ascending=True)

            data_per_batch.append(item_batch)

        data_per_brand[brand] = data_per_batch

    return data_per_brand


def rolling_window(data_, window):
    shape = data_.shape[:-1] + (data_.shape[-1] - window + 1, window)
    strides = data_.strides + (data_.strides[-1],)
    return np.lib.stride_tricks.as_strided(data_, shape=shape, strides=strides)


def calc_feature(item_: pd.DataFrame, feature_end: int, feature_range: int, split_num: int) -> np.array:
    """
    calc feature for each sample data
    :param item_: sample data
    :param feature_end: the end time to calc feature
    :param feature_range: feature calc range
    :param split_num: how many splits after splitting
    :return: feature array
    """
    feature_start = feature_end - feature_range
    # print('------------------')
    feature_slice = item_[feature_name_columns].iloc[feature_start: feature_end].values
    # print(len(feature_slice))
    # print(split_num)

    # shape = (SPLIT_NUM, FEATURE_RANGE / SPLIT_NUM, FEATURE_NUM)
    feature_slice = np.array(np.vsplit(feature_slice, split_num))
    # print(feature_slice.shape)
    # shape = (5, SPLIT_NUM, FEATURE_NUM)
    # 比如，feature前80个都是均值，在这80个里面，被分为了SPLIT_NUM段，每一段都是FEATURE_NUM个features
    feature = np.concatenate([
        np.mean(feature_slice, axis=1).ravel(),
        np.std(feature_slice, axis=1).ravel(),
        calc_integral(feature_slice).ravel(),
        skew(feature_slice, axis=1).ravel(),
        kurtosis(feature_slice, axis=1).ravel(),
    ])

    # print(len(feature.ravel()))
    # print('------------------')
    return feature.ravel()


def calc_label(item_: pd.DataFrame, start: int, end: int) -> np.array:
    """
    calc label for each sample
    :param item_: sample data
    :param start: the start time to calc label
    :param end: the end time to calc label
    :return: a array with exactly 2 number: temperature of region 1 and temperature of region 2
    """
    return np.mean(item_[label_column].iloc[start: end].values, axis=0)


def calc_delta(item_: pd.DataFrame, start: int, end: int) -> np.array:
    """
    计算Label变化的Delta值
    :param item_:
    :param start:
    :param end:
    :return:
    """
    label_ = calc_label(item_, start, end)
    delta = label_ - item_[label_column].iloc[start]
    return delta.values


def generate_brand_produce_training_data(item_brand, brand_index, setting, one_hot_brand) -> \
        Tuple[np.array, np.array, np.array, np.array]:
    """
    generate training data and label for one brand in 'produce' stage
    this method is time consuming
    :param item_brand: the brand data to generate
    :param brand_index: brand index
    :param setting: the setting value
    :param one_hot_brand: brand one hot encode
    :return:
        brand_train_data: all training data for this brand, shape=(N, M),
            which N denotes the number of training data, M denotes the number of feature
        brand_train_label: all training label for this brand, shape=(N, 2)
        brand_delta: delta info
        brand_mapping: mapping info
    """
    brand_train_data = []
    brand_train_label = []
    brand_delta = []
    brand_mapping = []

    for batch_index, item_batch in enumerate(item_brand):
        logging.info("|----Generating training data for batches, progress: {}"
                     .format(str(batch_index + 1) + "/" + str(len(item_brand))))
        item_batch = item_batch[item_batch[WORD_STATUS] == WORD_STATUS_PRODUCE]
        item_batch = item_batch.reset_index(drop=True)
        length = len(item_batch)
        humidity = item_batch[HUMIDITY].values

        stable_index = np.abs(humidity - setting) < MODEL_CRITERION
        # No stable area in this batch
        if np.sum(stable_index) == 0 or len(stable_index) < STABLE_WINDOWS_SIZE:
            continue
        stable_index = np.sum(rolling_window(stable_index, STABLE_WINDOWS_SIZE), axis=1)
        stable_index = np.where(stable_index == STABLE_WINDOWS_SIZE)[0]

        range_start = REACTION_LAG + SETTING_LAG + LABEL_RANGE + FEATURE_RANGE
        range_end = length - STABLE_WINDOWS_SIZE

        for stable_start in stable_index:
            if stable_start < range_start or stable_start >= range_end:
                continue
            adjust_end = stable_start - REACTION_LAG - SETTING_LAG
            adjust_start = adjust_end - LABEL_RANGE
            # store feature
            auxiliary_ = item_batch[HUMIDITY].values.ravel()[
                         adjust_end: stable_start + STABLE_WINDOWS_SIZE: FURTHER_STEP]
            auxiliary_ = auxiliary_ - setting
            brand_train_data.append(
                np.concatenate([
                    calc_feature(
                        item_batch,
                        adjust_start,
                        FEATURE_RANGE,
                        SPLIT_NUM
                    ), auxiliary_, [setting], one_hot_brand
                ])
            )

            # store label
            brand_train_label.append(
                calc_label(
                    item_batch,
                    adjust_start,
                    adjust_end
                )
            )

            # store mapping info
            brand_mapping.append([
                brand_index,
                batch_index,
                adjust_start,
                adjust_end,
                stable_start
            ])

            # store delta value
            brand_delta.append(
                calc_delta(
                    item_batch,
                    adjust_start,
                    adjust_end
                )
            )

    brand_train_data = np.array(brand_train_data)
    brand_train_label = np.array(brand_train_label)
    brand_mapping = np.array(brand_mapping)
    brand_delta = np.array(brand_delta)

    return brand_train_data, brand_train_label, brand_delta, brand_mapping


def generate_all_training_data(data_per_brand: dict, criterion: dict, one_hot: dict) -> Tuple[
    np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    generate training data and label for all brand
    :param data_per_brand: all brand data
    :param criterion: criterion for each brand
    :param one_hot: one hot encode for each brand
    :return:
        train_data: all training data, shape=(N, M),
            which N denotes the number of training data, M denotes the number of feature
        train_label: all training label, shape=(N, 2)
        delta: delta info
        mapping: mapping info
    """
    train_data_list = []
    train_label_list = []
    delta_list = []
    mapping_list = []

    for brand_index, brand in enumerate(data_per_brand):
        logging.info("Generating training data for brands, current: {}, progress: {}"
                     .format(brand, str(brand_index + 1) + "/" + str(len(data_per_brand))))
        start = datetime.now()

        brand_train_data, brand_train_label, brand_delta, brand_mapping = generate_brand_produce_training_data(
            data_per_brand[brand],
            brand_index,
            criterion[brand],
            np.array(one_hot[brand])
        )
        train_data_list.append(brand_train_data)
        train_label_list.append(brand_train_label)
        delta_list.append(brand_delta)
        mapping_list.append(brand_mapping)

        logging.info('{} : {}'.format(brand, len(brand_train_data)))
        logging.info('time: {}'.format(datetime.now() - start))

    X_produce = concatenate(train_data_list)
    y_produce = concatenate(train_label_list)
    delta_produce = concatenate(delta_list)
    mapping_produce = concatenate(mapping_list)

    return train_test_split(X_produce, y_produce, mapping_produce, delta_produce, test_size=0.2, random_state=6)
