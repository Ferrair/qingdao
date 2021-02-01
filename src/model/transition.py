import logging

from src.config.config import *
from src.model.base import BasicModel
from src.utils.util import save_dict_to_txt, read_txt_to_dict


class TransitionModel(BasicModel):
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
    def calc_work_dry(flow, input_flow_humid, input_humid, output_humid):
        # flow 流量
        # input_flow_humid 来料水分
        # input_humid 烘丝入口水分
        # output_humid 烘丝出口水分
        # 计算脱水量
        # @牛工给的计算脱水量的方法
        work_dry = flow * ((100 - input_flow_humid) / (100 - input_humid)) * (
                (input_humid - output_humid) / (100 - output_humid))
        return work_dry

    def predict(self, brand: str, flow: int, flow_set: int,
                humid_before_drying_sum: int, humid_before_drying_cur: int, output_humid: int, last_temp_1: float,
                last_temp_2: float, standard_temp_1: float, standard_temp_2: float,
                humid_sum: float, recent_humid: float = None, recent_work_dry: float = None) -> list:
        """
        predict in head stage

        :param flow_set:
        :param recent_humid: 这个地方是从数据库读取到的最近10批次的入口水分值
        :param recent_work_dry: 这个地方是从数据库读取到的最近10批次的脱水量
        :param standard_temp_2: 二区标准工作点位
        :param humid_sum 累计量
        :param standard_temp_1: 一区标准工作点位
        :param output_humid: 出口水分
        :param flow: 流量
        :param humid_before_drying_sum: 烘前水分累计值
        :param humid_before_drying_cur: 烘前水分顺时值
        :param last_temp_1: 上一个时间点的一区温度
        :param last_temp_2: 上一个时间点的二区温度
        :param brand: 牌号
        :return: predicted value
        """
        ##################################
        #### 使用脱水量来继续计算
        if recent_work_dry is not None:
            current_work_dry = self.calc_work_dry(flow=flow, input_flow_humid=humid_before_drying_cur,
                                                  input_humid=humid_sum + humid_before_drying_cur,
                                                  output_humid=output_humid)
            logging.info(
                'work dry: flow={}, before_drying_humid={}, input_humid={}, output_humid={}, work_dry={}'.format(
                    flow,
                    humid_before_drying_cur,
                    humid_sum + humid_before_drying_cur,
                    output_humid, current_work_dry))

            last_temp_1 = standard_temp_1 + DRY_RATIO * (current_work_dry - recent_work_dry) * TEMP_RATIO

        ##################################
        else:
            last_temp_1 = float(self.stable_per_brand[brand][0] + self.ratio[brand][
                0] * humid_before_drying_cur * 1.1 + standard_temp_1)

        ##################################
        #### 使用脱水量来继续计算
        if recent_work_dry is not None:
            current_work_dry = self.calc_work_dry(flow=flow, input_flow_humid=humid_before_drying_cur,
                                                  input_humid=humid_sum + humid_before_drying_cur,
                                                  output_humid=output_humid)
            logging.info(
                'work dry: flow={}, before_drying_humid={}, input_humid={}, output_humid={}, work_dry={}'.format(
                    flow,
                    humid_before_drying_cur,
                    humid_sum + humid_before_drying_cur,
                    output_humid, current_work_dry))

            last_temp_2 = standard_temp_2 + DRY_RATIO * (current_work_dry - recent_work_dry) * TEMP_RATIO

        ##################################
        else:
            last_temp_2 = float(self.stable_per_brand[brand][1] + self.ratio[brand][
                1] * humid_before_drying_cur * 1.1 + standard_temp_2)

        return [last_temp_1, last_temp_2]
