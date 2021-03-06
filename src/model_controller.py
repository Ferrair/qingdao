# -*- coding:UTF-8 -*-
import logging
import warnings

from src.PythonDeviceControlLib.HSControl import *
from src.config.error_code import *
from src.manager.model_manager import load_current_model, Determiner
from src.data_processing.processing import *
from flask import Flask, jsonify, request
import pandas as pd
from src.config.config import *
from src.utils.util import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

determiner = Determiner()
one_hot = None
previous_value = dict()
environment = None
failure = False
brand_strict = True
DEFAULT_BRAND = 'Txy###'
previous_time_dict = {}

# TODO: not hard code here

temp1_criterion = {'Txy###': 135,
                   'TG####A': 140,
                   'HSX###': 135,
                   'TH####A': 135,
                   'DQMr##': 130,
                   'ThQD##A': 135,
                   'HsxY##': 127,
                   'HR####': 145}
temp2_criterion = {'Txy###': 120,
                   'TG####A': 125,
                   'HSX###': 135,
                   'TH####A': 121,
                   'DQMr##': 130,
                   'ThQD##A': 120,
                   'HsxY##': 127,
                   'HR####': 145}


def api_select_current_model_name():
    current_model_name = request.json['current_model_name']
    return current_model_name


def get_auxiliary() -> list:
    """
    get auxiliary list in test phase
    """
    return [0] * 7


def gen_dataframe(originals: list) -> pd.DataFrame:
    """
    将传过来的数据 构造成DataFrame
    :param originals:
    :return:
    """
    if not originals:
        return pd.DataFrame()
    columns = originals[0].keys()
    data = []
    for item in originals:
        data.append(list(item.values()))
    return pd.DataFrame(data, columns=columns)


def wrap_success(data):
    return jsonify({
        "data": data,
        "code": 0,
        "msg": 'success'
    })


def wrap_failure(code, msg):
    return jsonify({
        "data": None,
        "code": code,
        "msg": msg
    })


def format_originals(originals):
    new_originals = []
    for original in originals:
        new_original = {}
        for item in original:
            try:
                new_original[item['id']] = float(item['v'])
            except Exception as e:
                new_original[item['id']] = item['v']
        new_originals.append(new_original)
    return new_originals


def add_random(pred):
    # FOR TEST USE ONLY
    # SHOULD REMOVE IN PROD MODE
    import random
    return [pred[0] + (random.random() - 0.5) / 10, pred[1] + (random.random() - 0.5) / 10]


def check_features_correct(features, originals):
    try:
        if len(originals) == FEATURE_RANGE:
            columns = list(originals[0].keys())
            data = [list(item.values()) for item in originals]
            computed_features = calc_feature_lr(pd.DataFrame(data, columns=columns), SPLIT_NUM)
            if not np.array_equal(features, computed_features):
                # logging.error('Flink Features process might not correct.')
                return False, computed_features
    except Exception as e:
        logging.error(e)
        return False, features
    return True, features


def _predict(originals, features, time_dict):
    # pred_start_time = time_dict.get('pred_start_time', 0)
    # kafka_time = time_dict.get('kafka_time', 0)
    sample_time = time_dict.get('sample_time', 0)

    # Check Feature
    # 判断Flink计算的特征是否正确
    is_correct, features = check_features_correct(features, originals)

    if len(originals) == 0:
        logging.error('len(originals) == 0')
        return wrap_failure(PARAMETERS_ERROR, 'len(originals) == 0')

    current_data = originals[len(originals) - 1]
    brand = current_data[BRADN]
    batch = current_data[BATCH]

    # # 检查顺序
    # global previous_time_dict
    # if sample_time < previous_time_dict.get(brand, 0):
    #     logging.error('sample_time: {} < previous_time: {}'.format(sample_time, previous_time_dict.get(brand)))
    #     return wrap_failure(PARAMETERS_ERROR,
    #                         'sample_time: {} < previous_time: {}'.format(sample_time, previous_time_dict.get(brand)))
    # previous_time_dict[brand] = sample_time

    if brand not in one_hot.keys():
        if brand_strict:
            logging.info('our model cannot handle new brand: ' + brand)
            return wrap_failure(NEW_BRAND_ERROR, 'our model cannot handle new brand: ' + brand)
        else:
            brand = DEFAULT_BRAND

    # len = 1650
    if len(features) != 0 and len(features) != (len(feature_name_columns) * 5 * SPLIT_NUM):
        return wrap_failure(PARAMETERS_ERROR, 'len(features) should equals {}, current: {}'.format(
            len(feature_name_columns) * 5 * SPLIT_NUM, len(features)))

    # check nan in features
    if sum(np.isnan(features)) > 0:
        return wrap_failure(PARAMETERS_ERROR, 'features contains nan')

    features = np.concatenate([features, get_auxiliary(), [criterion[brand]], one_hot[brand]])
    df = gen_dataframe(originals)

    logging.info('Start pred with len(features) = {}, len(originals) = {}'.format(len(features), len(originals)))

    try:
        pred = determiner.dispatch(df=df, features=features)
        logging.info('counter: {} -- Pred before adjust: {}, {}, HUMID: {}'
                     .format(determiner.counter, pred[0], pred[1], current_data[HUMID_AFTER_DRYING]))

        # 只有在生产阶段，才做这些操作
        if determiner.produce_flag:
            pred = adjust(pred, [x[HUMID_AFTER_DRYING] for x in originals], criterion[brand])
            logging.info('Pred after adjust self: {}, {}'.format(pred[0], pred[1]))
            try:
                n = int(determiner.adjust_params.get("n"))
                m = int(determiner.adjust_params.get("m"))
                k = float(determiner.adjust_params.get("k"))
                s = float(determiner.adjust_params.get("s"))
                x = float(np.mean(df[HUMID_AFTER_DRYING].values[-m:]) - criterion[brand])
                logging.info('Feedback: {}, {}, {}, {}, {}, {}'.format(determiner.counter, n, m, x, k, s))
                if int(determiner.counter) % n == 0:
                    pred[0] += float(x * k * s)
                    pred[1] += float(x * k * s)
                    logging.info('Pred after feedback {}, {}'.format(pred[0], pred[1]))
            except Exception as e:
                logging.error('Feedback error: {}'.format(e))

        pred = add_random(pred)
        ######
        # clip
        max_1 = int(determiner.adjust_params.get("max_1"))
        max_2 = int(determiner.adjust_params.get("max_2"))
        min_1 = float(determiner.adjust_params.get("min_1"))
        min_2 = float(determiner.adjust_params.get("min_2"))

        pred[0] = np.clip(pred[0], min_1, max_1)
        pred[1] = np.clip(pred[1], min_2, max_2)

        pred = clip_last(pred, float(current_data[TEMP1]), float(current_data[TEMP2]))
        ######

        logging.info('Pred after all: {}, {} ---- REAL: {}, {}'.format(pred[0], pred[1],
                                                                       current_data[TEMP1],
                                                                       current_data[TEMP2]))

    except Exception as e:
        logging.error(e)
        # TODO
        # 模型报错，需要发送通知
        return wrap_failure(MODEL_ERROR, 'Predict failure: {}'.format(e))

    result = {
        'brand': brand,  # str
        'batch': batch,  # str
        'tempRegion1': pred[0],  # float
        'tempRegion2': pred[1],  # float
        'time': sample_time,  # 采样数据的采样时间 # float
        'version': 'v2020.08.12',
        'debug_info': gen_debug_info(current_data)
    }
    logging.info('Pred success: {}, {}'.format(pred[0], pred[1]))
    logging_pred_in_disk(result)
    return wrap_success(result)


@app.route('/api/predict', methods=["POST"])
def predict_api():
    try:

        data = request.get_json()
        pred_start_time = int(time.time() * 1000)
        sample_time = data.get('time', 0)
        kafka_time = data.get('kafka_time', 0)
        features = data.get('features', [])
        originals = data.get('originals', [])
        originals = format_originals(originals)
        return _predict(originals, features, {
            'pred_start_time': pred_start_time,
            'sample_time': sample_time,
            'kafka_time': kafka_time,
        })
    except Exception as e:
        logging.error(e)
        return wrap_failure(UNKNOWN_ERROR, 'Unknown error {}'.format(e))


def logging_pred_in_disk(s):
    """
    写Logs，只写预测的结果
    """
    path = ROOT_PATH + '/logs/'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'pred_log.txt', 'a', buffering=1024 * 10) as f:
        f.write(str(get_current_time()) + ' ---- ' + str(s) + '\n')


def gen_debug_info(current_data):
    debug_info = {}

    if determiner.head_flag:
        debug_info['stage'] = 'head'
        debug_info['stage_timer'] = determiner.head_model.timer
    elif determiner.transition_flag:
        debug_info['stage'] = 'transition'
    elif determiner.produce_flag:
        debug_info['stage'] = 'produce'
    elif determiner.tail_flag:
        debug_info['stage'] = 'tail'
        debug_info['stage_timer'] = determiner.tail_model.timer
    else:
        debug_info['stage'] = 'unknown'

    debug_info['current_time'] = int(time.time() * 1000)  # 当前时间
    debug_info['flow'] = current_data[FLOW]
    debug_info['temp1_setting'] = current_data[TEMP1]
    debug_info['temp2_setting'] = current_data[TEMP2]
    debug_info['temp1_current'] = current_data[TEMP1_CURRENT]
    debug_info['temp2_current'] = current_data[TEMP2_CURRENT]
    debug_info['humid_out'] = current_data[HUMID_AFTER_DRYING]
    debug_info['humid_in'] = current_data[HUMID_BEFORE_DRYING]
    debug_info['work_state2'] = current_data[WORK_STATUS2]

    return debug_info


def logging_in_disk(s):
    """
    写Logs，所有的Logs都写到Disk里面去了
    """
    pass
    # with open('../logs/all_log.txt', 'a', buffering=1024 * 10) as f:
    #     f.write(str(get_current_time()) + ' ---- ' + str(s) + '\n')


@app.route('/api/manual_reset', methods=["POST"])
def manual_reset_api():
    warnings.warn("manual_reset_api is deprecated", DeprecationWarning)
    data = request.get_json()
    T1 = data.get('T1', 135)
    T2 = data.get('T2', 120)
    if environment == Environment.TEST:
        res = reset_test([str(T1), str(T2)])
        logging.info(res)
        return res
    elif environment == Environment.PROD:
        res = reset_prod([str(T1), str(T2)])
        logging.info(res)
        return res
    else:
        return jsonify("Fail")


@app.route('/api/manual_set', methods=["POST"])
def manual_set_api():
    warnings.warn("manual_set_api is deprecated", DeprecationWarning)
    data = request.get_json()
    try:
        T1 = data['T1']
        T2 = data['T2']
    except Exception as e:
        logging.error('Please set T1 and T2')
        return jsonify('Please set T1 and T2')

    global failure
    failure = True
    if environment == Environment.TEST:
        res = set_test([str(T1), str(T2)])
        logging.info(res)
        return res
    elif environment == Environment.PROD:
        res = set_prod([str(T1), str(T2)])
        logging.info(res)
        return res
    else:
        return jsonify("Fail")


@app.route('/api/healthz')
def healthz_api():
    return wrap_success('OK')


@app.route('/api/get_environment')
def get_environment_api():
    warnings.warn("get_environment is deprecated", DeprecationWarning)
    return jsonify(environment)


@app.route('/api/get_strict_mode')
def get_strict_mode_api():
    warnings.warn("get_strict_mode_api is deprecated", DeprecationWarning)
    return jsonify(brand_strict)


@app.route('/api/change_strict_mode')
def change_strict_mode_api():
    warnings.warn("change_strichange_strict_mode_apideprecated", DeprecationWarning)
    strict_mode = request.args.get("strict_mode")
    global brand_strict
    brand_strict = strict_mode
    return jsonify('OK')


# 模拟异常产生
@app.route('/api/mock_failure', methods=["GET"])
def mock_failure_api():
    warnings.warn("mock_failure_api is deprecated", DeprecationWarning)
    global failure
    failure = True
    return jsonify('mocked failure is trigger. {}'.format(failure))


# 恢复生产
@app.route('/api/reset_prod', methods=["GET"])
def reset_prod_api():
    warnings.warn("reset_prod_api is deprecated", DeprecationWarning)
    global failure
    failure = False
    return jsonify('reset_prod is trigger. {}'.format(failure))


@app.route('/api/change_env')
def change_env_api():
    warnings.warn("change_env_api is deprecated", DeprecationWarning)
    env = request.args.get("env")
    if env != Environment.NONE and env != Environment.PROD and env != Environment.TEST:
        return jsonify('Error')
    global environment
    environment = env

    save_config('env', environment)
    return jsonify('OK')


@app.route('/api/load_model_config')
def load_model_config_api():
    return jsonify({'window_size': FEATURE_RANGE, 'block_size': int(FEATURE_RANGE / SPLIT_NUM)})
    # stage = request.args.get("stage")
    # if stage == 'produce':
    # return jsonify({'window_size': FEATURE_RANGE, 'block_size': int(FEATURE_RANGE / SPLIT_NUM)})
    # elif stage == 'transition':
    #    return jsonify({'window_size': TRANSITION_FEATURE_RANGE,
    #                    'block_size': int(TRANSITION_FEATURE_RANGE / TRANSITION_SPLIT_NUM)})
    # else:
    #    raise Exception('param error')


@app.route('/api/load_temp_criterion')
def load_temp_criterion_api():
    return jsonify({'temp1_criterion': temp1_criterion, 'temp2_criterion': temp2_criterion})


if __name__ == '__main__':
    create_dir(MODEL_SAVE_DIR)
    one_hot = read_txt_to_dict(MODEL_SAVE_DIR + load_current_model('one-hot-brands'))
    environment = read_config('env')
    print('Current model: ', load_current_model('produce').split('/')[0])
    print('Current env: ', environment)
    app.run(host='0.0.0.0', port=5000)
