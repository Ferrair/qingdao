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

    def predict(self, brand: str, flow: int, humid_after_cut: int, last_temp_1: float, last_temp_2: float) -> list:
        """
        predict in head stage
        :param flow: 流量
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
        if self.timer == self.range_1_lag:
            last_temp_1 = float(self.stable_per_brand[brand][0] + self.ratio[brand] * humid_after_cut)

        # region 2
        if self.timer == self.range_2_lag:
            last_temp_2 = float(self.stable_per_brand[brand][1] + self.ratio[brand] * humid_after_cut)

        self.timer += 1
        return [last_temp_1, last_temp_2]
