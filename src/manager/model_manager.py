import json
import logging
import os
from queue import Queue

import pymssql
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_absolute_error
from src.config.config import *
from src.config.db_setup import *
from src.model.head import HeadModel
from src.model.lr_model import LRModel
from src.model.tail import TailModel
from src.model.transition import TransitionModel
from src.utils.util import read_config, save_config, get_current_time

from joblib import dump


def load_all_model_dir() -> list:
    return sorted(os.listdir(MODEL_SAVE_DIR), reverse=True)


def load_latest_model_dir() -> str:
    return load_all_model_dir()[0]


def make_new_model_dir(current_time):
    os.mkdir(MODEL_SAVE_DIR + "/" + current_time)


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
        return param
    else:
        raise Exception('param MUST in [produce, transition, head, one-hot-brands], now is ' + param)


def humid_stable(original_humid: list, setting: float) -> bool:
    """
    连续 20 条数据出口水分与设定值误差不大于 0.15, 则认为出口水分已稳定
    :param original_humid: 输入的出口水分数据
    :param setting: 出口水分设定值
    :return:
    """
    try:
        if len(original_humid) < 20:
            return False

        original_humid = original_humid[-20:]
        original_humid_diff = np.array([abs(float(i) - setting) for i in original_humid])
        if np.any(original_humid_diff > 0.15):
            return False

        return True
    except Exception as e:
        logging.exception('humid_stable error: {}'.format(e))
        return False


def train_and_save_model(X: np.array,
                         X_test: np.array,
                         y: np.array,
                         y_test: np.array,
                         brand_name_list: list,
                         current_time: str,
                         stage='produce'):
    """
    训练生产阶段模型
    :param y_test:
    :param X_test:
    :param brand_name_list:
    :param X:
    :param y:
    :param stage
    :param current_time
    :return:
    """
    new_model = LRModel()
    logging.info("Training {} ...".format(stage))
    new_model.train(X, y)

    X_test_scaler = new_model.scaler.transform(X_test)
    pred = new_model.model.predict(X_test_scaler)
    mae = round(mean_absolute_error(y_test[:, :2], pred[:, :2]), 3)
    logging.info('mae: {}'.format(mae))
    mae = 0.10

    model_filename = MODEL_SAVE_DIR + load_latest_model_dir() + "/" + current_time + "#{}".format(stage) + "#" + str(
        mae)
    one_hot_filename = MODEL_SAVE_DIR + load_latest_model_dir() + "/" + current_time + "#one-hot-brands.txt"

    dump(new_model.model, model_filename + '.joblib')
    dump(new_model.scaler, model_filename + '.pkl')

    ont_hot_dict = to_one_hot(brand_name_list)
    dump(ont_hot_dict, one_hot_filename)


def to_one_hot(brand_name_list_):
    rslt = {}
    for index, brand_name in enumerate(brand_name_list_):
        arr = [0] * len(brand_name_list_)
        arr[index] = 1
        rslt[brand_name] = arr
    return rslt


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
        self.transition_counter = 0

        # 计算生产状态的
        self.produce_flag = False

        # 计算尾料的
        self.tail_flag = False

        self.q = Queue()

        self.adjust_params = {}

        self.counter = 0

        self.last_batch = None

        self.recent_batch_humid = None

        self.recent_work_dry = None

        self.standard_temp = {}

    @staticmethod
    def read_recent_info(brand, kpi_name):
        try:
            sql = """
                SELECT TOP(10) kpiaverage FROM BigDataBSS.dbo.Fact_Quality_Performance
                WHERE brandcode = '{brand}' and kpiname = '{kpi_name}'
                order by workorderstartdate desc
            """.format(brand=brand, kpi_name=kpi_name)

            conn = pymssql.connect(server=QP_DB_HOST,
                                   user=QP_DB_USER,
                                   password=QP_DB_PWD,
                                   database='BigDataBSS')
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            recent_humid = []
            for row in rows:
                recent_humid.append(float(row[0]))
            logging.info('read_recent_info info {}'.format(float(np.mean(recent_humid))))
            return float(np.mean(recent_humid))
        except Exception as e:
            logging.exception('read_recent_info error {}'.format(e))
            return None

    @staticmethod
    def read_standard_temp(brand):
        try:
            sql = """
                    SELECT 
                    Zone1_Pre_heating, Zone2_Pre_heating, Zone1_Work_heating, Zone2_Work_heating  
                    FROM ML.dbo.MODEL_ZS_HS_PARA
                    WHERE Brand = '{brand}'
                """.format(brand=brand)

            conn = pymssql.connect(server=FEEDBACK_DB_HOST,
                                   user=FEEDBACK_DB_USER,
                                   password=FEEDBACK_DB_PWD,
                                   database='ML')
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            standard_temp = {}
            for row in rows:
                standard_temp = {
                    's1_pre': row[0],
                    's2_pre': row[1],
                    's1': row[2],
                    's2': row[3]
                }
            logging.info('read_standard_temp info {}'.format(standard_temp))
            return standard_temp
        except Exception as e:
            logging.exception('read_standard_temp error {}'.format(e))
            return None

    def read_adjust_params(self, brand):
        # default values
        n, m, k, s, min_1, max_1, min_2, max_2 = 1, 30, 10, 0.4, 130, 140, 130, 140
        try:
            sql = """
                   SELECT FeedbackN, FeedbackM, FeedbackK, FeedbackS, Tagcode, Min, Max
                   FROM ML.dbo.FeedbackValue WHERE Process = 'LD5' AND Batch = '{}'
            """.format(brand)
            conn = pymssql.connect(server=FEEDBACK_DB_HOST,
                                   user=FEEDBACK_DB_USER,
                                   password=FEEDBACK_DB_PWD,
                                   database='ML')
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                n = int(row[0])
                m = int(row[1])
                k = float(row[2])
                s = float(row[3])
                if row[4] == STANDARD_TEMP_1:
                    min_1 = float(row[5])
                    max_1 = float(row[6])
                if row[4] == STANDARD_TEMP_2:
                    min_2 = float(row[5])
                    max_2 = float(row[6])
        except Exception as e:
            logging.exception('read_adjust_params error: {}'.format(e))

        self.adjust_params = {
            "n": n,
            "m": m,
            "k": k,
            "s": s,
            "max_1": max_1,
            "min_1": min_1,
            "max_2": max_2,
            "min_2": min_2,
        }

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

    def update_model(self):
        if self.produce_model is not None:
            self.produce_model.load(MODEL_SAVE_DIR + load_current_model('produce'))
        # if self.transition_model is not None:
        #    self.transition_model.load(MODEL_SAVE_DIR + load_current_model('transition'))

    def init_model(self, next_range_1: int, next_range_2: int):
        logging.info('init Model')
        self.head_model = HeadModel()
        self.tail_model = TailModel(next_range_1, next_range_2)
        self.transition_model = TransitionModel()
        self.produce_model = LRModel()

        self.head_model.load(CONFIG_PATH + load_current_model('head'))
        self.transition_model.load(CONFIG_PATH + load_current_model('head'))

        self.produce_model.load(MODEL_SAVE_DIR + load_current_model('produce'))

        # 计算头料的
        self.head_flag = False

        # 过渡状态
        self.transition_flag = False
        self.transition_counter = 0

        # 计算生产状态的
        self.produce_flag = False

        # 计算尾料的
        self.tail_flag = False

        self.humid_after_cut = []  # 清空
        self.cut_half_full_counter = 0

        self.q = Queue()

        self.counter = 0

    def dispatch(self, df: pd.DataFrame, produce_features) -> list:
        """
        :param df: 一个Windows长度的数据，数组最后一个点的数据为当前时刻的数据
        :param produce_features: 特征：只有produce才会使用
        非常重要的一个的方法，根据数据来判断使用那个模型，并进行预测，然后输出结果
        :return:
        """
        len_ = len(df)
        if len_ < MIN_DATA_NUM:
            raise Exception('len(originals) MUST >= {}'.format(MIN_DATA_NUM))
        current_data = df.iloc[len_ - 1]  # 最新的一条数据
        last_data = df.iloc[len_ - 2]  # 上一秒一条数据
        current_batch = self.last_batch
        logging.info('Load current batch: {}, flow: {}, flag: {}, {}, {}, {}'.format(current_batch,
                                                                                     current_data[FLOW],
                                                                                     self.head_flag,
                                                                                     self.transition_flag,
                                                                                     self.produce_flag,
                                                                                     self.tail_flag))
        current_brand = current_data[BRADN]

        # current_batch = None
        try:
            # 计算切后水分，只选取 5000 叶丝线暂存柜半满后的三分钟的数据
            # 改为：切后水分仪计算到时间范围：以入口水分大于17后的60S开始计时，持续到半满后的2分钟
            # 5H.5H.LD5_KL2226_InputMoisture

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
                # save_config('current_batch', current_batch)
                self.last_batch = current_batch

                err, standard_obj = self.read_standard(current_brand, DEFAULT_STANDARD_1, DEFAULT_STANDARD_2)
                if not err:
                    logging.info('Get standard success: {}'.format(standard_obj))
                    self.init_model(standard_obj.get('standard_1'), standard_obj.get('standard_2'))
                else:
                    logging.exception('Get standard error: {}'.format(err))
                    self.init_model(DEFAULT_STANDARD_1, DEFAULT_STANDARD_2)

                # 在每个批次开始的时候读取反馈控制
                self.read_adjust_params(brand=current_brand)

                KPI_HUMID = '五千烘丝入口水分'
                KPI_WORK_DRY = '五千滚筒烘丝脱水量'

                self.recent_batch_humid = self.read_recent_info(brand=current_brand, kpi_name=KPI_HUMID)
                self.recent_work_dry = self.read_recent_info(brand=current_brand, kpi_name=KPI_WORK_DRY)
                self.standard_temp = self.read_standard_temp(brand=current_brand)

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

            logging.info('Checkpoint 1 --- Check Stage Finish')

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

            logging.info('Checkpoint 2 --- Check Stage Finish')

            if self.head_flag:
                logging.info('Current in Head Model.')
                try:
                    # 根据牛工建议选用前1/3最大的水分humid_after_cut进行计算平均值，去除调较小的水分:降序排列
                    humid_after_cut_ = sorted(self.humid_after_cut, reverse=True)
                    humid_after_cut_clip = humid_after_cut_[:int(len(humid_after_cut_) / 3)]
                    humid_after_cut_float = sum(humid_after_cut_clip) / len(humid_after_cut_clip)
                    logging.info('humid_after_cut_float: {}'.format(humid_after_cut_float))
                except Exception as e:
                    try:
                        humid_after_cut_float = sum(self.humid_after_cut) / len(self.humid_after_cut)
                    except ZeroDivisionError as e:
                        logging.info(
                            'humid_after_cut_float ZeroDivisionError: {}, {}'.format(
                                sum(self.humid_after_cut),
                                len(self.humid_after_cut))
                        )
                        humid_after_cut_float = 19

                try:
                    humid_before_drying = list(self.q.queue)
                    humid_before_drying_float = sum(humid_before_drying) / len(humid_before_drying)
                except ZeroDivisionError as e:
                    logging.info('humid_before_drying ZeroDivisionError: {}, {}'.format(sum(list(self.q.queue)),
                                                                                        len(list(self.q.queue))))
                    humid_before_drying_float = 17

                logging.info('Start Head')
                try:
                    pred = self.head_model.predict(brand=current_data[BRADN],
                                                   flow_set=float(current_data[FLOW_SET]),
                                                   flow=float(current_data[FLOW]),
                                                   recent_humid=self.recent_batch_humid,
                                                   output_humid=current_data[HUMID_AFTER_DRYING_SETTING],
                                                   recent_work_dry=self.recent_work_dry,
                                                   humid_sum=current_data[HUMID_MOIST_INC],
                                                   humid_before_drying_sum=humid_before_drying_float,
                                                   humid_before_drying_cur=current_data[HUMID_BEFORE_DRYING],
                                                   humid_after_cut_sum=humid_after_cut_float,
                                                   standard_temp_2=float(self.standard_temp.get('s2')),
                                                   standard_temp_1=float(self.standard_temp.get('s1')),
                                                   last_temp_1=float(current_data[TEMP1]),
                                                   last_temp_2=float(current_data[TEMP2]))
                except Exception as e:
                    pred = [float(current_data[TEMP1]), float(current_data[TEMP2])]
                    logging.exception('head fail: {}'.format(e))

                logging.info('Head timer: {}'.format(self.head_model.timer))
                return list(pred)

            if self.transition_flag:
                self.transition_counter += 1
                logging.info('Current in Transition Model.')
                brand = current_data[BRADN]
                # try:
                #     humid_after_cut_float = sum(self.humid_after_cut_sum) / len(self.humid_after_cut_sum)
                # except ZeroDivisionError as e:
                #     logging.info(
                #         'ZeroDivisionError: {}, {}'.format(sum(self.humid_after_cut_sum), len(self.humid_after_cut_sum)))
                #     humid_after_cut_float = 17

                humid_before_drying = list(self.q.queue)
                try:
                    humid_before_drying_float = sum(humid_before_drying) / len(humid_before_drying)
                except ZeroDivisionError as e:
                    logging.info('ZeroDivisionError: {}, {}'.format(sum(humid_before_drying), len(humid_before_drying)))
                    humid_before_drying_float = 17
                # 暂时使用Head模型，增加了下惩罚项

                humid_use = humid_before_drying_float

                logging.info(
                    'Transition info 1: {}, {}, {}, {}'.format(self.head_model.stable_per_brand[brand][0],
                                                               self.head_model.ratio[brand][0],
                                                               humid_use,
                                                               self.standard_temp.get('s1')))
                logging.info(
                    'Transition info 2: {}, {}, {}, {}'.format(self.head_model.stable_per_brand[brand][1],
                                                               self.head_model.ratio[brand][1],
                                                               humid_use,
                                                               self.standard_temp.get('s2')))

                last_temp_1 = float(
                    self.head_model.stable_per_brand[brand][0] + self.head_model.ratio[brand][0]
                    * humid_use + float(self.standard_temp.get('s1')))
                last_temp_2 = float(
                    self.head_model.stable_per_brand[brand][1] + self.head_model.ratio[brand][1]
                    * humid_use + float(self.standard_temp.get('s2')))

                try:
                    pred = self.transition_model.predict(
                        brand=current_data[BRADN],
                        flow_set=float(current_data[FLOW_SET]),
                        flow=float(current_data[FLOW]),
                        recent_humid=self.recent_batch_humid,
                        output_humid=current_data[HUMID_AFTER_DRYING_SETTING],
                        recent_work_dry=self.recent_work_dry,
                        humid_sum=current_data[HUMID_MOIST_INC],
                        humid_before_drying_sum=humid_before_drying_float,
                        humid_before_drying_cur=current_data[HUMID_BEFORE_DRYING],
                        standard_temp_2=float(self.standard_temp.get('s2')),
                        standard_temp_1=float(self.standard_temp.get('s1')),
                        last_temp_1=float(current_data[TEMP1]),
                        last_temp_2=float(current_data[TEMP2])
                    )
                    logging.info(
                        'last_temp_1/2: {}, {}, pred: {}, {}'.format(last_temp_1, last_temp_2, pred[0], pred[1]))
                    return [last_temp_1 * 0.5 + pred[0] * 0.5, last_temp_2 * 0.5 + pred[1] * 0.5]
                except Exception as e:
                    logging.exception('transition fail: {}'.format(e))

                return [last_temp_1, last_temp_2]

            if self.produce_flag:
                logging.info('Current in Produce Model.')
                logging.info('features shape: {}'.format(produce_features.shape))
                self.counter += 1
                pred = self.produce_model.predict(produce_features)
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
            logging.exception(e)
            # save_config('current_batch', 'error')
            self.last_batch = None
            return [float(current_data[TEMP1]), float(current_data[TEMP2])]
