import logging
import os
import pandas as pd
import numpy as np
from src.config.config import *
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
            # MACOS 会存在 .DS_Store
            if file.startswith('.'):
                continue
            if os.path.splitext(file)[0].split('#')[1] == 'produce':
                return current_dir + "/" + os.path.splitext(file)[0]
    elif param in ['head', 'one-hot-brands']:
        return current_dir + "/" + current_dir + '#' + param
    else:
        raise Exception('param MUST in [produce, transition, head, one-hot-brands], now is ' + param)


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
        :param df: 一个Windows长度的数据，数组最后一个点的数据为当前时刻的数据
        :param features: 特征：只有produce才会使用
        非常重要的一个的方法，根据数据来判断使用那个模型，并进行预测，然后输出结果
        :return:
        """
        len_ = len(df)
        if len_ < MIN_DATA_NUM:
            raise Exception('len(originals) MUST >= {}'.format(MIN_DATA_NUM))
        current_data = df.iloc[len_ - 1]  # 最新的一条数据
        last_data = df.iloc[len_ - 2]  # 上一秒一条数据
        current_batch = read_config('current_batch')
        # current_batch = None

        try:
            # 计算切后水分，只选取 5000叶丝线暂存柜半满 后的三分钟的数据
            if current_data[CUT_HALF_FULL]:
                self.cut_half_full_flag = True
            if self.cut_half_full_flag and len(self.humid_after_cut) < HUMID_AFTER_CUT_RANGE:
                self.humid_after_cut.append(current_data[HUMID_AFTER_CUT])

            # 一个批次的开始
            if not current_batch or current_batch != current_data[BATCH]:
                current_batch = current_data[BATCH]
                save_config('current_batch', current_batch)
                # TODO 需要更换
                # self.init_model(current_data[WARM_TEMP1], current_data[WARM_TEMP2])
                self.init_model(130, 115)

            # 当前点的流量增长到了 2000 --> HeadModel
            if last_data[FLOW] < FLOW_LIMIT < current_data[FLOW]:
                self.head_flag = True
                self.produce_flag = False
                self.tail_flag = False

            # 当前点有了出口水分，并且头料也进行一段时间了 --> ProductModel
            # 或者直接已经就是生产状态
            if current_data[HUMID_AFTER_CUT] > HUMID_EPSILON and self.head_model.timer > HEAD_MAX_TIME:
                self.head_flag = False
                self.produce_flag = True
                self.tail_flag = False

            # 当前点的流量减少到了 2000 --> TailModel
            if last_data[FLOW] > FLOW_LIMIT > current_data[FLOW]:
                self.head_flag = False
                self.produce_flag = False
                self.tail_flag = True

            # 兜底策略
            if not self.head_flag and not self.produce_flag and not self.tail_flag:
                if current_data[WORK_STATUS] == 32:
                    self.head_flag = False
                    self.produce_flag = True
                    self.tail_flag = False
                elif current_data[WORK_STATUS] == 16 or current_data[WORK_STATUS] == 8:
                    self.head_flag = True
                    self.produce_flag = False
                    self.tail_flag = False
                elif current_data[WORK_STATUS] == 16 or current_data[WORK_STATUS] == 64:
                    self.head_flag = False
                    self.produce_flag = False
                    self.tail_flag = True
                else:
                    raise Exception('Find invalid work status.')

            if self.head_flag:
                logging.info('Current in Head Model.')
                pred = self.head_model.predict(brand=current_data[BRADN], flow=current_data[FLOW],
                                               humid_after_cut=sum(self.humid_after_cut) / len(self.humid_after_cut),
                                               last_temp_1=current_data[TEMP1], last_temp_2=current_data[TEMP2])
                logging.info('Head timer: {}'.format(self.head_model.timer))
                return list(pred)

            if self.produce_flag:
                # TODO: transition model 没有使用
                logging.info('Current in Produce Model.')
                pred = self.produce_model.predict(features)
                return list(pred.ravel())

            if self.tail_flag:
                logging.info('Current in Tail Model.')
                finish, pred = self.tail_model.predict(flow=current_data[FLOW],
                                                       last_temp_1=current_data[TEMP1],
                                                       last_temp_2=current_data[TEMP2])
                # TODO: 逻辑还需要在处理下
                # if finish:
                #    save_config('current_batch', None)
                logging.info('Tail timer: {}'.format(self.tail_model.timer))
                return list(pred)
        except Exception as e:
            logging.error(e)
            save_config('current_batch', None)
            return [current_data[TEMP1], current_data[TEMP2]]
