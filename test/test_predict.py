import logging

from src.PythonDeviceControlLib.HSControl import *
from src.config.error_code import *
from src.data_processing.processing import *
from src.config.config import *
from src.manager.model_manager import Determiner, load_current_model
from src.model_controller import wrap_failure, brand_strict, DEFAULT_BRAND, gen_dataframe, get_auxiliary, \
    temp1_criterion, temp2_criterion, add_random, wrap_success
from src.data_processing.processing import FEATURE_RANGE, calc_feature_lr, SPLIT_NUM
from src.utils.util import *

determiner = Determiner()
one_hot = read_txt_to_dict(MODEL_SAVE_DIR + load_current_model('one-hot-brands'))


def _predict(originals_, features_, pred_start_time=0, sample_time=0):
    current_data = originals_[len(originals_) - 1]
    brand = current_data[BRADN]
    batch = current_data[BATCH]

    if brand not in one_hot.keys():
        if brand_strict:
            logging.info('our model cannot handle new brand: ' + brand)
            return wrap_failure(NEW_BRAND_ERROR, 'our model cannot handle new brand: ' + brand)
        else:
            brand = DEFAULT_BRAND
    # len = 1650
    if len(features_) != 0 and len(features_) != (len(feature_name_columns) * 5 * SPLIT_NUM):
        return wrap_failure(PARAMETERS_ERROR, 'len(features) should equals {}, current: {}'.format(
            len(feature_name_columns) * 5 * SPLIT_NUM, len(features_)))

    # check nan in features
    if sum(np.isnan(features_)) > 0:
        return wrap_failure(PARAMETERS_ERROR, 'features contains nan')

    features_ = np.concatenate([features_, get_auxiliary(), [criterion[brand]], one_hot[brand]])
    df = gen_dataframe(originals_)

    logging.info('Start pred with len(features) = {}, len(originals) = {}'.format(len(features_), len(originals_)))

    try:
        pred = determiner.dispatch(df=df, features=features_)
        logging.info('Pred before adjust: {}, {}'.format(pred[0], pred[1]))
        # 只有在生产阶段，才做这些操作
        if determiner.produce_flag:
            pred = adjust(pred, [x[HUMID_AFTER_DRYING] for x in originals_], criterion[brand])
            pred = clip(pred, temp1_criterion[brand], temp2_criterion[brand])
        pred = clip_last(pred, current_data[TEMP1], current_data[TEMP2])
        pred = add_random(pred)
        logging.info('Pred after adjust: {}, {}'.format(pred[0], pred[1]))
        pred_end_time = int(time.time() * 1000)
    except Exception as e:
        logging.error(e)
        return wrap_failure(MODEL_ERROR, 'Predict failure: {}'.format(e))

    result = {
        'brand': brand,  # str
        'batch': batch,  # str
        'tempRegion1': pred[0],  # float
        'tempRegion2': pred[1],  # float
        'time': sample_time,  # 采样数据的采样时间 # float
        'upstream_consume': pred_start_time - sample_time,  # 所有上游任务消耗的时间 # float
        'pred_consume': pred_end_time - pred_start_time,  # 预测消耗的时间 # float
        'plc_consume': 0,  # call plc 消耗的时间 # float
        'version': 'v2020.08.12',
    }
    logging.info('Pred success: {}, {}'.format(pred[0], pred[1]))
    return result


filename = '/Users/bytedance/Downloads/hs0902/MODEL_ZS_5K_LD5_HS_Dev20-09-02-11-40-00.1599018000781.log'
originals = []
values = []
with open(filename) as f:
    lines = f.readlines()

for line in lines:
    line = json.loads(line)
    value_dict = {}
    for value in line.get('values'):
        value_dict[value.get('id')] = value.get('v')
    values.append(value_dict)
    del value_dict

for i in range(0, len(values) - FEATURE_RANGE):
    originals.append(values[i:i + FEATURE_RANGE])

rslt = []
for original in originals:
    columns = list(original[0].keys())
    data = [list(item.values()) for item in original]
    t = calc_feature_lr(pd.DataFrame(data, columns=columns), SPLIT_NUM)
    rslt_ = _predict(original, t)
    rslt.append(rslt_)

print(rslt)
