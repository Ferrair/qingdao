from src.model.base import BasicModel
from src.utils.util import save_dict_to_txt, read_txt_to_dict
import numpy as np


class TailModel(BasicModel):
    def __init__(self, next_range_1: float, next_range_2: float, flow_min_limit: int = 2000, range_lag_1: int = 30,
                 range_lag_2: int = 90, rate: float = 0.25, plain_duration: int = 180, plain_temp: int = 100):
        """
        :param next_range_1: 下一次预热的温度
        :param next_range_2: 下一次预热的温度
        :param flow_min_limit: 流量低于什么值，就开始触发尾料
        :param range_lag_1: 过多少秒，一区开始递减
        :param range_lag_2: 过多少秒，二区开始递减
        :param rate: 递减/或者递增的rate
        :param plain_duration: 温度最终会到达100度，并持续一段时间
        :param plain_temp: 温度最终会到达100度

        一区温度基本是135，二区温度是120，然后 range_lag_2-range_lag_1 = 60S, 60S * rate 约等于 135 - 120，
        所以plain_duration时间对于两个区都是一样的。不需要进行区分
        """
        self.plain_temp = plain_temp
        self.next_range_2 = next_range_2
        self.next_range_1 = next_range_1
        self.plain_duration = plain_duration
        self.rate = rate
        self.range_lag_2 = range_lag_2
        self.range_lag_1 = range_lag_1
        self.flow_min_limit = flow_min_limit

        self.plain_timer = 0
        self.timer = 0
        self.batch = None

    def train(self, init_per_brand: dict, stable_per_brand: dict):
        pass

    def save(self, saved_path: str):
        """
        Save model to disk
        :param saved_path: path and filename to save model and scaler
        """
        pass

    def load(self, loaded_path: str):
        """
        load model from disk
        :param loaded_path: path and filename to load model and scaler
        """
        pass

    def check_model_state(self):
        pass

    def predict(self, batch: str, flow: int, last_temp_1: float, last_temp_2: float) -> list:
        """
        predict in head stage
        :param last_temp_2: 上次的二区温度
        :param last_temp_1: 上次的一区温度
        :param batch: batch name
        :param flow: 流量
        :return: predicted value
        """
        if flow >= self.flow_min_limit:
            raise Exception(
                'flow({}) <= self.flow_min_limit({}), please do not use tail model.'.format(flow, self.flow_min_limit))

        if not self.batch:
            self.batch = batch
            self.timer = 0
            self.plain_timer = 0

        self.timer += 1

        # 1. 递减的过程
        if self.timer >= self.range_lag_1 and last_temp_1 - self.plain_temp > self.rate \
                and self.plain_timer < self.plain_duration:
            last_temp_1 -= self.rate
        if self.timer >= self.range_lag_2 and last_temp_2 - self.plain_temp > self.rate \
                and self.plain_timer < self.plain_duration:
            last_temp_2 -= self.rate

        # 3. 递增的过程
        if self.plain_timer == self.plain_duration:
            if last_temp_1 - self.next_range_1 < self.rate:
                last_temp_1 += self.rate
            if last_temp_2 - self.next_range_2 < self.rate:
                last_temp_2 += self.rate
            return [last_temp_1, last_temp_2]

        # 2. 降到最低点，持续一段时间
        if last_temp_2 - self.plain_temp <= self.rate and last_temp_1 - self.plain_temp <= self.rate:
            self.plain_timer += 1

        return [last_temp_1, last_temp_2]
