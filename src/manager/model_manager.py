import os
import pandas as pd
import numpy as np
from src.config.config import MODEL_SAVE_DIR
from src.model.head import HeadModel
from src.model.lr_model import LRModel
from src.model.tail import TailModel
from src.utils.util import read_config, save_config


def load_all_model_dir() -> list:
    return sorted(os.listdir(MODEL_SAVE_DIR), reverse=True)


def load_latest_model_dir() -> str:
    return load_all_model_dir()[0]


def load_current_model(param: str) -> str:
    current_dir = load_latest_model_dir()
    if param in ['produce', 'transition']:
        for file in os.listdir(MODEL_SAVE_DIR + current_dir):
            if os.path.splitext(file)[0].split('#')[1] == 'produce':
                return current_dir + "/" + os.path.splitext(file)[0]
    elif param in ['head', 'one-hot-brands']:
        return current_dir + "/" + current_dir + '#' + param
    else:
        raise Exception('param MUST in [produce, transition, head, one-hot-brands], now is ' + param)


FLOW = '5H.5H.LD5_CK2222_TbcLeafFlowSH'  # '瞬时流量'
BATCH = '6032.6032.LD5_YT603_2B_YS2ROASTBATCHNO'  # '批次'
BRADN = '6032.6032.LD5_YT603_2B_YS2ROASTBRAND'  # '牌号'
HUMID_AFTER_CUT = '6032.6032.LD5_TM2222A_CUTOUTMOISTURE'  # '切丝后出口水分'
TEMP1 = '5H.5H.LD5_KL2226_BucketTemp1SP'  # '一区温度设定值'
TEMP2 = '5H.5H.LD5_KL2226_BucketTemp2SP'  # '二区温度设定值'
HUMID_AFTER_DRYING = '5H.5H.LD5_KL2226_TT1LastMoisPV'  # '烘丝后出口水分'
CUT_HALF_FULL = '6032.6032.LD5_2220_GP2220STATUS3'  # '5000叶丝线暂存柜半满'

# TODO：这个2个地方需要修改
WARM_TEMP1 = '一区预热'
WARM_TEMP2 = '二区预热'

HUMID_AFTER_CUT_RANGE = 180  # 选取这么多时间的切丝后出口水分平均值
FLOW_LIMIT = 2000  # 流量判断
MIN_DATA_NUM = 10  # 最少的数据限制
HUMID_EPSILON = 0.1  # 低于这个出口水分，几乎就为0，为了防止误差的
HEAD_MAX_TIME = 300  # 头料阶段最大时间，大于这个时间就当做生产状态了


class Determiner:

    def __init__(self) -> None:
        super().__init__()
        self.head_model = None
        self.tail_model = None
        self.transition_model = None
        self.produce_model = None

        # 计算下个批次预热的
        # self.next_range_1 = next_range_1
        # self.next_range_2 = next_range_2

        # 计算5000叶丝线暂存柜半满的
        self.humid_after_cut = []
        self.cut_half_full_flag = False

        # 计算头料的
        self.head_flag = False

        # 计算生产状态的
        self.produce_flag = False

        # 计算尾料的
        self.tail_flag = False

    def init_model(self, next_range_1: int, next_range_2: int):
        self.head_model = HeadModel()
        self.tail_model = TailModel(next_range_1, next_range_2)
        self.produce_model = LRModel()
        self.transition_model = LRModel()

        self.head_model.load(MODEL_SAVE_DIR + load_current_model('head'))
        self.produce_model.load(MODEL_SAVE_DIR + load_current_model('produce'))
        self.transition_model.load(MODEL_SAVE_DIR + load_current_model('transition'))

    def dispatch(self, df: pd.DataFrame, features: np.array) -> list:
        """
        TODO: 需要测试
        :param df: 一个Windows长度的数据，数组最后一个点的数据为当前时刻的数据
        :param features: 特征：只有produce才会使用
        非常重要的一个的方法，根据数据来判断使用那个模型，并进行预测，然后输出结果
        :return:
        """
        len_ = len(df)
        if len_ <= MIN_DATA_NUM:
            return []
        current_data = df.iloc[len_ - 1]  # 最新的一条数据
        last_data = df.iloc[len_ - 2]  # 上一秒一条数据
        current_batch = read_config('current_batch')

        # 计算切后水分，只选取 5000叶丝线暂存柜半满 后的三分钟的数据
        if current_data[CUT_HALF_FULL]:
            self.cut_half_full_flag = True
        if self.cut_half_full_flag and len(self.humid_after_cut) < HUMID_AFTER_CUT_RANGE:
            self.humid_after_cut.append(current_data[HUMID_AFTER_CUT])

        # 一个批次的开始
        if not current_batch or current_batch != current_data[BATCH]:
            current_batch = current_data[BATCH]
            save_config('current_batch', current_batch)
            self.init_model(current_data[WARM_TEMP1], current_data[WARM_TEMP2])

        # 当前点的流量增长到了 2000 --> HeadModel
        if last_data[FLOW] < FLOW_LIMIT < current_data[FLOW]:
            self.head_flag = True
            self.produce_flag = False
            self.tail_flag = False

        # 当前点有了出口水分，并且头料也进行一段时间了 --> ProductModel
        # TODO: 使用PLC里面点位信息来进行判断
        if current_data[HUMID_AFTER_CUT] > HUMID_EPSILON and self.head_model.timer > HEAD_MAX_TIME:
            self.head_flag = False
            self.produce_flag = True
            self.tail_flag = False

        # 当前点的流量减少到了 2000 --> TailModel
        if last_data[FLOW] > FLOW_LIMIT > current_data[FLOW]:
            self.head_flag = False
            self.produce_flag = False
            self.tail_flag = True

        if self.head_flag:
            pred = self.head_model.predict(brand=current_data[BRADN], flow=current_data[FLOW],
                                           humid_after_cut=sum(self.humid_after_cut) / len(self.humid_after_cut),
                                           last_temp_1=current_data[TEMP1], last_temp_2=current_data[TEMP2])
            return list(pred)

        if self.produce_flag:
            # TODO: transition model 没有使用
            pred = self.head_model.predict(features)
            return list(pred)

        if self.tail_flag:
            pred = self.tail_model.predict(flow=current_data[FLOW],
                                           last_temp_1=current_data[TEMP1],
                                           last_temp_2=current_data[TEMP2])
            return list(pred)
