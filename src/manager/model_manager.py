import json
import logging
import os
from queue import Queue

import pandas as pd
import numpy as np
import requests
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


def humid_stable(original_humid: list, setting: float) -> bool:
    """
    连续 10 条数据出口水分与设定值误差不大于 0.1, 则认为出口水分已稳定
    :param original_humid: 输入的出口水分数据
    :param setting: 出口水分设定值
    :return:
    """
    try:
        if len(original_humid) < 10:
            return False

        original_humid = original_humid[-10:]
        original_humid_diff = np.array([abs(float(i) - setting) for i in original_humid])
        if np.any(original_humid_diff > 0.1):
            return False

        return True
    except Exception as e:
        logging.error('humid_stable error: {}'.format(e))
        return False


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
        self.cut_half_full_counter = 0

        # 计算头料的
        self.head_flag = False

        # 过渡状态
        self.transition_flag = False

        # 计算生产状态的
        self.produce_flag = False

        # 计算尾料的
        self.tail_flag = False

        self.q = Queue()

    @classmethod
    def read_standard(cls, brand, default_1, default_2):
        try:
            # 请求预热的配置
            body = {
                "BrandCode": brand,
                "WorkstageCode": "LD5",
                "TagReads": [
                    STANDARD_TEMP_1,
                    STANDARD_TEMP_2
                ]
            }
            res = requests.post(CONFIG_URL, json=body)
            logging.info('Standard: {} {}'.format(res.status_code, res.text))
            if (res.status_code / 100) == 2:
                json_obj = json.loads(res.text)
                rows = json_obj.get('data').get('Rows')
                standard_1 = default_1
                standard_2 = default_2
                for row in rows:
                    if row.get('TagRead') == STANDARD_TEMP_1:
                        standard_1 = float(row.get('ParmSet')) - 3
                    if row.get('TagRead') == STANDARD_TEMP_2:
                        standard_2 = float(row.get('ParmSet')) - 3
                return None, {
                    'standard_1': standard_1,
                    'standard_2': standard_2
                }
            else:
                return res.text, None
        except Exception as e:
            return str(e), None

    def init_model(self, next_range_1: int, next_range_2: int):
        self.head_model = HeadModel()
        self.tail_model = TailModel(next_range_1, next_range_2)
        self.produce_model = LRModel()
        self.transition_model = LRModel()

        self.head_model.load(MODEL_SAVE_DIR + load_current_model('head'))
        self.produce_model.load(MODEL_SAVE_DIR + load_current_model('produce'))
        self.transition_model.load(MODEL_SAVE_DIR + load_current_model('transition'))

        # 计算头料的
        self.head_flag = False

        # 过渡状态
        self.transition_flag = False

        # 计算生产状态的
        self.produce_flag = False

        # 计算尾料的
        self.tail_flag = False

        self.humid_after_cut = []  # 清空
        self.cut_half_full_counter = 0

        self.q = Queue()

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
        logging.info('Load current batch: {}'.format(current_batch))
        current_brand = current_data[BRADN]

        # current_batch = None
        try:
            # 计算切后水分，只选取 5000 叶丝线暂存柜半满后的三分钟的数据
            # 改为：切后水分仪计算到时间范围：以入口水分大于17后的60S开始计时，持续到半满后的2分钟
            # 5H.5H.LD5_KL2226_InputMoisture
            # TODO 优化为queue的形式
            if float(current_data[HUMID_AFTER_CUT]) > 17.5 and self.cut_half_full_counter < 180:
                self.humid_after_cut.append(float(current_data[HUMID_AFTER_CUT]))
            if current_data[CUT_HALF_FULL]:
                self.cut_half_full_counter += 1

            self.q.put(float(current_data[HUMID_BEFORE_DRYING]))
            if self.q.qsize() > MAX_BEFORE_HUMID_SIZE:
                self.q.get()

            # 一个批次的开始
            if not current_batch or current_batch != current_data[BATCH]:
                current_batch = current_data[BATCH]
                save_config('current_batch', current_batch)

                err, standard_obj = self.read_standard(current_brand, DEFAULT_STANDARD_1, DEFAULT_STANDARD_2)
                if not err:
                    logging.info('Get standard success: {}'.format(standard_obj))
                    self.init_model(standard_obj.get('standard_1'), standard_obj.get('standard_2'))
                else:
                    logging.error('Get standard error: {}'.format(err))
                    self.init_model(DEFAULT_STANDARD_1, DEFAULT_STANDARD_2)

            logging.info('Checkpoint 1 --- Check Stage')
            # 当前点的流量增长到了 2000 --> HeadModel
            if float(last_data[FLOW]) < FLOW_LIMIT < float(current_data[FLOW]):
                self.head_flag = True
                self.transition_flag = False
                self.produce_flag = False
                self.tail_flag = False

            # 当前点有了出口水分，并且未进入生产阶段 --> TransitionModel
            if float(current_data[HUMID_AFTER_DRYING]) > HUMID_EPSILON and self.head_flag:
                self.head_flag = False
                self.transition_flag = True
                self.produce_flag = False
                self.tail_flag = False

            # 当前就是生产阶段，或者出口水分已稳定 --> ProductModel
            if self.produce_flag is True or humid_stable(list(df[HUMID_AFTER_DRYING].values),
                                                         float(criterion[current_brand])):
                self.head_flag = False
                self.transition_flag = False
                self.produce_flag = True
                self.tail_flag = False

            # 流量小于2000，并且之前状态是生产状态 --> TailModel
            if FLOW_LIMIT > float(current_data[FLOW]) and self.produce_flag:
                self.head_flag = False
                self.transition_flag = False
                self.produce_flag = False
                self.tail_flag = True

            logging.info('Checkpoint 2 --- Check Stage Default')
            # 兜底策略
            if not self.head_flag and not self.produce_flag and not self.tail_flag and not self.transition_flag:
                if int(current_data[WORK_STATUS1]) == 32:
                    self.head_flag = False
                    self.transition_flag = False
                    self.produce_flag = True
                    self.tail_flag = False
                elif int(current_data[WORK_STATUS1]) == 16 or int(current_data[WORK_STATUS1]) == 8:
                    self.head_flag = True
                    self.transition_flag = False
                    self.produce_flag = False
                    self.tail_flag = False
                elif int(current_data[WORK_STATUS1]) == 16 or int(current_data[WORK_STATUS1]) == 64:
                    self.head_flag = False
                    self.transition_flag = False
                    self.produce_flag = False
                    self.tail_flag = True
                else:
                    raise Exception('Invalid work status. So we will use last 2 temp as current temp. FLOW: {}'.format(
                        current_data[FLOW]))

            logging.info('Checkpoint 3 --- Check Stage Finish')
            if self.head_flag:
                logging.info('Current in Head Model.')
                try:
                    humid_after_cut = sum(self.humid_after_cut) / len(self.humid_after_cut)
                except ZeroDivisionError as e:
                    logging.info(
                        'ZeroDivisionError: {}, {}'.format(sum(self.humid_after_cut), len(self.humid_after_cut)))
                    humid_after_cut = 17
                pred = self.head_model.predict(brand=current_data[BRADN],
                                               flow=float(current_data[FLOW]),
                                               humid_after_cut=humid_after_cut,
                                               standard_temp_2=float(current_data[STANDARD_TEMP_2]),
                                               standard_temp_1=float(current_data[STANDARD_TEMP_1]),
                                               last_temp_1=float(current_data[TEMP1]),
                                               last_temp_2=float(current_data[TEMP2]))
                logging.info('Head timer: {}'.format(self.head_model.timer))
                return list(pred)

            if self.transition_flag:
                logging.info('Current in Transition Model.')
                brand = current_data[BRADN]
                input_humid = list(self.q.queue)
                try:
                    input_humid = sum(input_humid) / len(input_humid)
                except ZeroDivisionError as e:
                    logging.info('ZeroDivisionError: {}, {}'.format(sum(input_humid), len(input_humid)))
                    input_humid = 17
                # 暂时使用Head模型，增加了下惩罚项

                last_temp_1 = float(
                    self.head_model.stable_per_brand[brand][0] + self.head_model.ratio[brand][0]
                    * input_humid * 1.1 + float(current_data[STANDARD_TEMP_1]))
                last_temp_2 = float(
                    self.head_model.stable_per_brand[brand][1] + self.head_model.ratio[brand][1]
                    * input_humid * 1.1 + float(current_data[STANDARD_TEMP_2]))
                return [last_temp_1, last_temp_2]

            if self.produce_flag:
                logging.info('Current in Produce Model.')
                pred = self.produce_model.predict(features)
                return list(pred.ravel())

            if self.tail_flag:
                logging.info('Current in Tail Model.')
                finish, pred = self.tail_model.predict(flow=float(current_data[FLOW]),
                                                       last_temp_1=float(current_data[TEMP1]),
                                                       last_temp_2=float(current_data[TEMP2]))
                # TODO: 逻辑还需要在处理下
                # if finish:
                #    save_config('current_batch', None)
                logging.info('Tail timer: {}, is_finish: {}'.format(self.tail_model.timer, finish))
                return list(pred)
        except Exception as e:
            logging.error(e)
            save_config('current_batch', None)
            return [float(current_data[TEMP1]), float(current_data[TEMP2])]
