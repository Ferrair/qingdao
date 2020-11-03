import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

from src.utils.util import name_list_2_plc_list

feature_name_columns = ['最终烟丝含水实际值', '烘丝出口温度', '瞬时流量', '累计流量',
                        '罩压力反馈值', '罩压力实际值', '桶温区1反馈值', '桶温区1实际值', '桶温区1设定值', '桶温区2反馈值',
                        '桶温区2实际值', '桶温区2设定值', '工艺气体温度反馈值', '工艺气体温度实际值',
                        '入口烟丝水分', '回潮机水分增加', '最终水分PID控制反馈值',
                        '罩压力PID控制反馈值', '工艺气速度反馈值', '工艺气速度实际值', '工艺气速度设定值', '工艺气体温度设定值',
                        '区域1为滚筒加热正向控制除水', '区域2为滚筒加热正向控制除水',
                        '最终烟丝含水设定值', '桶温区1阀后蒸汽压力', '桶温区2阀后蒸汽压力', '工作点脱水', '脱水速度',
                        '工艺蒸汽流量实际值', '工艺蒸汽流量设定值', 'SIROX阀后蒸汽压力', 'Sirox出口物料温度']

feature_plc_columns = name_list_2_plc_list(feature_name_columns)

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


def calc_feature_lgbm(item_: pd.DataFrame) -> np.array:
    """
    测试阶段用于计算特征的
    calc feature for each sample data
    :param item_: sample data
    :return: feature array
    """
    return item_[feature_plc_columns].values.ravel()


def calc_feature_lr(item_: pd.DataFrame, split_num: int) -> np.array:
    """
    测试阶段用于计算特征的
    calc feature for each sample data
    :param item_: sample data
    :param split_num: how many splits after splitting
    :return: feature array
    """
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
    bound = 0.3
    pred[0] = np.clip(pred[0], last_temp_1 - bound, last_temp_1 + bound)
    pred[1] = np.clip(pred[1], last_temp_2 - bound, last_temp_2 + bound)
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
    ratio = 1.2
    if np.all(original_humid_diff > 0.05):
        pred[0] += np.sum(original_humid_diff) * ratio
        pred[1] += np.sum(original_humid_diff) * ratio
    if np.all(original_humid_diff < 0.05):
        pred[0] -= np.sum(original_humid_diff) * ratio
        pred[1] -= np.sum(original_humid_diff) * ratio
    return pred
