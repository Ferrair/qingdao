import logging

import numpy as np

from src.config.config import *
from src.model.base import BasicModel
from src.utils.util import save_dict_to_txt, read_txt_to_dict


class HeadModel(BasicModel):
    def __init__(self, range_1_lag: int = 30, range_2_lag: int = 120, flow_min_limit: int = FLOW_LIMIT):
        self.flow_min_limit = flow_min_limit
        self.ratio = {}
        self.stable_per_brand = {}
        self.timer = 0

        self.range_2_lag = range_2_lag
        self.range_1_lag = range_1_lag

    def train(self, ratio: dict, stable_per_brand: dict):
        self.ratio = ratio
        self.stable_per_brand = stable_per_brand

    def save(self, saved_path: str):
        """
        Save model to disk
        :param saved_path: path and filename to save model and scaler
        """
        super().save(saved_path)
        self.check_model_state()
        head_model = {'ratio': self.ratio, 'stable_per_brand': self.stable_per_brand}
        save_dict_to_txt(saved_path, head_model)

    def load(self, loaded_path: str):
        """
        load model from disk
        :param loaded_path: path and filename to load model and scaler
        """
        super().load(loaded_path)
        head_model = read_txt_to_dict(loaded_path)
        self.ratio = head_model['ratio']
        self.stable_per_brand = head_model['stable_per_brand']

    def check_model_state(self):
        if self.ratio is None:
            raise Exception('No available ratio.')
        if self.stable_per_brand is None:
            raise Exception('No available stable_per_brand.')

    @staticmethod
    def calc_work_dry(flow, after_cut_humid, input_humid, output_humid):
        # flow 流量
        # after_cut_humid 来料水分
        # input_humid 烘丝入口水分
        # output_humid 烘丝出口水分
        # 计算脱水量
        # @牛工给的计算脱水量的方法
        work_dry = flow * ((100 - after_cut_humid) / (100 - input_humid)) * (
                (input_humid - output_humid) / (100 - output_humid))
        return work_dry

    def predict(self, brand: str, flow: int, humid_after_cut: int, humid_before_drying: int, output_humid: int,
                last_temp_1: float, last_temp_2: float, standard_temp_1: float, standard_temp_2: float,
                humid_sum: float, recent_humid: float = None, recent_work_dry: float = None) -> list:
        """
        predict in head stage

        :param recent_humid: 这个地方是从数据库读取到的最近10批次的入口水分值
        :param recent_work_dry: 这个地方是从数据库读取到的最近10批次的脱水量
        :param standard_temp_2: 二区标准工作点位
        :param humid_sum 累计量
        :param standard_temp_1: 一区标准工作点位
        :param output_humid: 出口水分
        :param flow: 流量
        :param humid_before_drying: 烘前水分
        :param humid_after_cut: 切丝后出口水分
        :param last_temp_1: 上一个时间点的一区温度
        :param last_temp_2: 上一个时间点的二区温度
        :param brand: 牌号
        :return: predicted value
        """
        if flow < self.flow_min_limit:
            raise Exception(
                'flow({}) <= self.flow_min_limit({}), please do not use head model.'.format(flow, self.flow_min_limit))
        if not self.ratio or not self.stable_per_brand:
            raise Exception('No available head model, please train a new model or load from disk.')

        # if not self.batch or self.batch != batch:
        #     self.batch = batch
        #     self.timer = 0

        # region 1
        logging.info(
            'Head info 1: {}, {}, {}, {}, {}, {}'.format(self.stable_per_brand[brand][0],
                                                         self.ratio[brand][0],
                                                         humid_after_cut,
                                                         humid_sum,
                                                         humid_before_drying,
                                                         standard_temp_1))
        logging.info(
            'Head info 2: {}, {}, {}, {}, {}, {}'.format(self.stable_per_brand[brand][1],
                                                         self.ratio[brand][1],
                                                         humid_after_cut,
                                                         humid_sum,
                                                         humid_before_drying,
                                                         standard_temp_2))

        # self.calc_work_dry(flow=flow, after_cut_humid=humid_after_cut, input_humid=humid_sum, output_humid=output_humid)

        if self.timer >= self.range_1_lag:
            # if recent_humid is not None:
            #     if self.timer >= self.range_1_lag + 120:
            #         last_temp_1 = standard_temp_1 + 2 * (humid_after_cut - recent_humid)
            #     else:
            #         last_temp_1 = standard_temp_1 + 2 * (humid_after_cut - recent_humid)
            ##################################
            #### 使用脱水量来继续计算
            if recent_work_dry is not None:
                if self.timer >= self.range_1_lag + 120:
                    current_work_dry = self.calc_work_dry(flow=flow, after_cut_humid=humid_before_drying,
                                                          input_humid=humid_sum + humid_before_drying,
                                                          output_humid=output_humid)
                    logging.info(
                        'work dry: flow={}, after_cut_humid={}, input_humid={}, output_humid={}, work_dry={}'.format(
                            flow,
                            humid_before_drying,
                            humid_sum + humid_before_drying,
                            output_humid, current_work_dry))

                    last_temp_1 = standard_temp_1 + 0.12 * (current_work_dry - recent_work_dry)
                else:
                    current_work_dry = self.calc_work_dry(flow=flow, after_cut_humid=humid_after_cut,
                                                          input_humid=humid_sum + humid_after_cut,
                                                          output_humid=output_humid)
                    logging.info(
                        'work dry: flow={}, after_cut_humid={}, input_humid={}, output_humid={}, work_dry={}'.format(
                            flow,
                            humid_before_drying,
                            humid_sum + humid_before_drying,
                            output_humid, current_work_dry))

                    last_temp_1 = standard_temp_1 + 0.12 * (current_work_dry - recent_work_dry)
            ##################################
            else:
                if self.timer >= self.range_1_lag + 120:
                    last_temp_1 = float(self.stable_per_brand[brand][0] + self.ratio[brand][
                        0] * humid_before_drying * 1.1 + standard_temp_1)
                else:
                    last_temp_1 = float(self.stable_per_brand[brand][0] + self.ratio[brand][
                        0] * humid_after_cut * 1.1 + standard_temp_1)

        # region 2
        if self.timer >= self.range_2_lag:
            # if recent_humid is not None:
            #     if self.timer >= self.range_2_lag + 120:
            #         last_temp_2 = standard_temp_2 + 2 * (humid_before_drying - recent_humid)
            #     else:
            #         last_temp_2 = standard_temp_2 + 2 * (humid_after_cut - recent_humid)

            ##################################
            #### 使用脱水量来继续计算
            if recent_work_dry is not None:
                if self.timer >= self.range_2_lag + 120:
                    current_work_dry = self.calc_work_dry(flow=flow, after_cut_humid=humid_before_drying,
                                                          input_humid=humid_sum + humid_before_drying,
                                                          output_humid=output_humid)
                    logging.info(
                        'work dry: flow={}, after_cut_humid={}, input_humid={}, output_humid={}, work_dry={}'.format(
                            flow,
                            humid_before_drying,
                            humid_sum + humid_before_drying,
                            output_humid, current_work_dry))

                    last_temp_2 = standard_temp_2 + 0.12 * (current_work_dry - recent_work_dry)
                else:
                    current_work_dry = self.calc_work_dry(flow=flow, after_cut_humid=humid_before_drying,
                                                          input_humid=humid_sum + humid_before_drying,
                                                          output_humid=output_humid)
                    logging.info(
                        'work dry: flow={}, after_cut_humid={}, input_humid={}, output_humid={}, work_dry={}'.format(
                            flow,
                            humid_before_drying,
                            humid_sum + humid_before_drying,
                            output_humid, current_work_dry))

                    last_temp_2 = standard_temp_2 + 0.12 * (current_work_dry - recent_work_dry)
            ##################################
            else:
                if self.timer >= self.range_2_lag + 120:
                    last_temp_2 = float(self.stable_per_brand[brand][1] + self.ratio[brand][
                        1] * humid_before_drying * 1.1 + standard_temp_2)
                else:
                    last_temp_2 = float(self.stable_per_brand[brand][1] + self.ratio[brand][
                        1] * humid_after_cut * 1.1 + standard_temp_2)

        self.timer += 1
        return [last_temp_1, last_temp_2]
