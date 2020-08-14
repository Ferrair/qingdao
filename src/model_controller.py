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

# TODO: not hard code here
criterion = {'Txy###': 12.699999999999994,
             'TG####A': 12.493271237066992,
             'HSX###': 13.80000000000001,
             'TH####A': 12.49285817787605,
             'DQMr##': 13.799999999999997,
             'ThQD##A': 12.5,
             'HsxY##': 13.5,
             'HR####': 12.8}
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
            new_original[item['id']] = item['v']
        new_originals.append(new_original)
    return new_originals


def add_random(pred):
    # FOR TEST USE ONLY
    import random
    return [pred[0] + random.random() / 2, pred[1] + random.random() / 2]


def _predict():
    data = request.get_json()
    pred_start_time = int(time.time() * 1000)
    sample_time = data.get('time', 0)
    features = data.get('features', [])
    originals = data.get('originals', [])
    originals = format_originals(originals)

    if len(originals) == 0:
        logging.error('len(originals) == 0')
        return wrap_failure(PARAMETERS_ERROR, 'len(originals) == 0')

    current_data = originals[len(originals) - 1]
    brand = current_data[BRADN]
    batch = current_data[BATCH]

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
        logging.info('Pred before adjust: {}, {}'.format(pred[0], pred[1]))
        # 只有在生产阶段，才做这些操作
        if determiner.produce_flag:
            pred = adjust(pred, [x[HUMID_AFTER_DRYING] for x in originals], criterion[brand])
            pred = clip(pred, temp1_criterion[brand], temp2_criterion[brand])
        pred = clip_last(pred, current_data[TEMP1], current_data[TEMP2])
        pred = add_random(pred)
        logging.info('Pred after adjust: {}, {}'.format(pred[0], pred[1]))
        pred_end_time = int(time.time() * 1000)
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
        'upstream_consume': pred_start_time - sample_time,  # 所有上游任务消耗的时间 # float
        'pred_consume': pred_end_time - pred_start_time,  # 预测消耗的时间 # float
        'plc_consume': 0,  # call plc 消耗的时间 # float
        'version': 'v2020.08.12'
    }
    logging.info('Pred success: {}, {}'.format(pred[0], pred[1]))
    logging_pred_in_disk(result)
    return wrap_success(result)


@app.route('/api/predict', methods=["POST"])
def predict_api():
    try:
        return _predict()
    except Exception as e:
        logging.error(e)
        return wrap_failure(999, 'Unknown error {}'.format(e))


def logging_pred_in_disk(s):
    """
    写Logs，只写预测的结果
    """
    path = ROOT_PATH + '/logs/'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'pred_log.txt', 'a', buffering=1024 * 10) as f:
        f.write(str(get_current_time()) + ' ---- ' + str(s) + '\n')


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
    return jsonify(wrap_success('OK'))


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
