import json
import os
import numpy as np
from src.config.config import *


def get_value_by_id(values, id_):
    for value in values:
        if value.get('id') == id_:
            return value.get('v')
    return None


def mean(dict_):
    new_dict_ = {}
    for key, values in dict_.items():
        new_dict_[key] = np.mean(list(values))
    return new_dict_


PATH_2 = '/Users/bytedance/Qihang/青岛/数据/hs07/MODEL_ZS_5K_LD5_HS/'
PATH_3 = '/Users/bytedance/Qihang/青岛/数据/hs08/MODEL_ZS_5K_LD5_HS/'

file_list_2 = sorted(os.listdir(PATH_2))
file_list_3 = sorted(os.listdir(PATH_3))

temp1_per_batch = {}
temp2_per_batch = {}

temp1_sta_per_batch = {}
temp2_sta_per_batch = {}

humid_per_batch = {}

for file in file_list_2:
    if file.startswith('.'):
        continue
    try:
        with open(PATH_2 + file) as f:
            lines = f.readlines()

            first_line = json.loads(lines[0].replace('\n', ''))
            last_line = json.loads(lines[-1].replace('\n', ''))
            first_flow = float(get_value_by_id(first_line.get('values'), FLOW_TOTAL))
            last_flow = float(get_value_by_id(last_line.get('values'), FLOW_TOTAL))
            # 这个文件没有生产，跳过吧
            if first_flow < 10 and last_flow < 10:
                continue
            batch = get_value_by_id(first_line.get('values'), BATCH)
            for line in lines:
                line = json.loads(line.replace('\n', ''))
                flow = float(get_value_by_id(line.get('values'), FLOW_TOTAL))

                temp1 = float(get_value_by_id(line.get('values'), TEMP1))
                temp2 = float(get_value_by_id(line.get('values'), TEMP2))
                humid = float(get_value_by_id(line.get('values'), HUMID_AFTER_CUT))
                temp1_sta = float(get_value_by_id(line.get('values'), STANDARD_TEMP_1))
                temp2_sta = float(get_value_by_id(line.get('values'), STANDARD_TEMP_2))
                # 只选取累计流量在[10,1000]之间的这段时间
                if 1000 > flow > 10:
                    if batch not in temp1_per_batch.keys():
                        temp1_per_batch[batch] = []
                        temp2_per_batch[batch] = []
                        temp1_sta_per_batch[batch] = []
                        temp2_sta_per_batch[batch] = []
                        humid_per_batch[batch] = []

                    temp1_per_batch[batch].append(temp1)
                    temp2_per_batch[batch].append(temp2)
                    temp1_sta_per_batch[batch].append(temp1_sta)
                    temp2_sta_per_batch[batch].append(temp2_sta)
                    humid_per_batch[batch].append(humid)
    except Exception as e:
        pass

for file in file_list_3:
    if file.startswith('.'):
        continue
    try:
        with open(PATH_3 + file) as f:
            lines = f.readlines()

            first_line = json.loads(lines[0].replace('\n', ''))
            last_line = json.loads(lines[-1].replace('\n', ''))
            first_flow = float(get_value_by_id(first_line.get('values'), FLOW_TOTAL))
            last_flow = float(get_value_by_id(last_line.get('values'), FLOW_TOTAL))
            # 这个文件没有生产，跳过吧
            if first_flow < 10 and last_flow < 10:
                continue
            batch = get_value_by_id(first_line.get('values'), BATCH)
            for line in lines:
                line = json.loads(line.replace('\n', ''))
                flow = float(get_value_by_id(line.get('values'), FLOW_TOTAL))

                temp1 = float(get_value_by_id(line.get('values'), TEMP1))
                temp2 = float(get_value_by_id(line.get('values'), TEMP2))
                humid = float(get_value_by_id(line.get('values'), HUMID_AFTER_CUT))
                temp1_sta = float(get_value_by_id(line.get('values'), STANDARD_TEMP_1))
                temp2_sta = float(get_value_by_id(line.get('values'), STANDARD_TEMP_2))
                # 只选取累计流量在[10, 1000]之间的这段时间
                if 1000 > flow > 10:
                    if batch not in temp1_per_batch.keys():
                        temp1_per_batch[batch] = []
                        temp2_per_batch[batch] = []
                        temp1_sta_per_batch[batch] = []
                        temp2_sta_per_batch[batch] = []
                        humid_per_batch[batch] = []

                    temp1_per_batch[batch].append(temp1)
                    temp2_per_batch[batch].append(temp2)
                    temp1_sta_per_batch[batch].append(temp1_sta)
                    temp2_sta_per_batch[batch].append(temp2_sta)
                    humid_per_batch[batch].append(humid)
    except Exception as e:
        pass
print(mean(temp1_per_batch))
print(mean(temp2_per_batch))
print(mean(temp1_sta_per_batch))
print(mean(temp2_sta_per_batch))
print(mean(humid_per_batch))
