# -*- coding:UTF-8 -*-
import logging

from src.PythonDeviceControlLib.HSControl import *
from src.manager.model_manager import load_current_model, Determiner
from src.data_processing.processing import *
from flask import Flask, jsonify, request
import pandas as pd
from src.config.config import MODEL_SAVE_DIR, Environment, ROOT_PATH
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


def train_val_model():
    pass


def validate(data_per_batch: pd.DataFrame) -> np.array:
    pass


def get_auxiliary() -> list:
    """
    get auxiliary list in test phase
    """
    return [0] * int((REACTION_LAG + SETTING_LAG + STABLE_WINDOWS_SIZE) / FURTHER_STEP)


def check_dim(current: int, required: int):
    """
    检查传入features的维度
    """
    if current != required:
        raise Exception('len(features) wrong, excepted=' + str(required) + ' current=' + str(current))


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


@app.route('/api/predict', methods=["POST"])
def predict_api():
    data = request.get_json()
    pred_start_time = int(time.time() * 1000)
    sample_time = data.get('time', 0)
    # storm_time = int(time.time() * 1000)
    # window_time = data.get('window_time', 0)
    # model_time = data.get('model_time', 0)
    # kafka_time = data.get('kafka_time', 0)
    # batch = data.get('batch', None)
    # brand = data.get('brand', None)
    features = data['features']
    originals = data.get('originals', [])
    if len(originals) == 0:
        logging.error('len(originals) == 0')
        return jsonify('len(originals) == 0')

    current_data = originals[len(originals) - 1]
    brand = current_data['6032.6032.LD5_YT603_2B_YS2ROASTBRAND']
    batch = current_data['6032.6032.LD5_YT603_2B_YS2ROASTBATCHNO']

    if brand not in one_hot.keys():
        if brand_strict:
            logging.info('our model cannot handle new brand: ' + brand)
            return jsonify('our model cannot handle new brand: ' + brand)
        else:
            brand = DEFAULT_BRAND

    features = np.concatenate([features, get_auxiliary(), [criterion[brand]], one_hot[brand]])
    df = gen_dataframe(originals)

    try:
        pred = determiner.dispatch(df=df, features=features)
        pred = adjust(pred, [x['5H.5H.LD5_KL2226_TT1LastMoisPV'] for x in originals], criterion[brand])
        pred = clip(pred, temp1_criterion[brand], temp2_criterion[brand])
        pred_end_time = int(time.time() * 1000)
    except Exception as e:
        logging.error(e)
        # TODO
        # 模型报错，需要发送通知
        return 'Error'

    if environment == Environment.PROD and not failure:
        try:
            roll_back = False
            res = set_prod([str(pred[0]), str(pred[1])])
            logging.info(res)
            logging_in_disk(res)
            for r in res:
                roll_back = roll_back or not r['IsSetSuccessful']
                if r['Address'] == '5H.5H.LD5_KL2226_TT1StandardTemp1':
                    previous_value['T1'] = r['PreviousValue']
                elif r['Address'] == '5H.5H.LD5_KL2226_TT1StandardTemp2':
                    previous_value['T2'] = r['PreviousValue']

            if roll_back:
                res = reset_prod([str(previous_value['T1']), str(previous_value['T2'])])
                logging.info(res)
                logging_in_disk(res)
        except Exception as e:
            logging.error(e)
            logging_in_disk(e)
            return 'Error'
    elif environment == Environment.TEST and not failure:
        try:
            roll_back = False
            res = set_test([str(pred[0]), str(pred[1])])
            logging.info(res)
            logging_in_disk(res)
            for r in res:
                roll_back = roll_back or not r['IsSetSuccessful']
                if r['Address'] == '5H.5H.LD5_KL2226_TT1StandardTemp1':
                    previous_value['T1'] = r['PreviousValue']
                elif r['Address'] == '5H.5H.LD5_KL2226_TT1StandardTemp2':
                    previous_value['T2'] = r['PreviousValue']

            if roll_back:
                res = reset_prod([str(previous_value['T1']), str(previous_value['T2'])])
                logging.info(res)
                logging_in_disk(res)
        except Exception as e:
            logging.error(e)
            logging_in_disk(e)
            return 'Error'
    elif environment == Environment.NONE:
        pass

    result = {
        'brand': brand,
        'batch': batch,
        'tempRegion1': pred[0],
        'tempRegion2': pred[1],
        'time': sample_time,  # 采样数据的采样时间
        'upstream_comsume': pred_start_time - sample_time,  # 所有上游任务消耗的时间
        'pred_consume': pred_end_time - pred_start_time,  # 预测消耗的时间
        'plc_consume': int(time.time() * 1000) - pred_end_time,  # call plc 消耗的时间
        'version': '1.2'
    }
    logging_pred_in_disk(result)
    return jsonify(result)


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


@app.route('/api/get_environment')
def test_api():
    return jsonify(environment)


@app.route('/api/get_strict_mode')
def get_strict_mode():
    return jsonify(brand_strict)


@app.route('/api/change_strict_mode')
def change_strict_mode():
    strict_mode = request.args.get("strict_mode")
    global brand_strict
    brand_strict = strict_mode
    return jsonify('OK')


# 模拟异常产生
@app.route('/api/mock_failure', methods=["GET"])
def mock_failure_api():
    global failure
    failure = True
    return jsonify('mocked failure is trigger. {}'.format(failure))


# 恢复生产
@app.route('/api/reset_prod', methods=["GET"])
def reset_prod_api():
    global failure
    failure = False
    return jsonify('reset_prod is trigger. {}'.format(failure))


@app.route('/api/change_env')
def change_env_api():
    env = request.args.get("env")
    if env != Environment.NONE and env != Environment.PROD and env != Environment.TEST:
        return jsonify('Error')
    global environment
    environment = env

    save_config('env', environment)
    return jsonify('OK')


@app.route('/api/load_model_config')
def api_load_model_config():
    stage = request.args.get("stage")
    if stage == 'produce':
        return jsonify({'window_size': FEATURE_RANGE, 'block_size': int(FEATURE_RANGE / SPLIT_NUM)})
    elif stage == 'transition':
        return jsonify({'window_size': TRANSITION_FEATURE_RANGE,
                        'block_size': int(TRANSITION_FEATURE_RANGE / TRANSITION_SPLIT_NUM)})
    else:
        raise Exception('param error')


if __name__ == '__main__':
    create_dir(MODEL_SAVE_DIR)
    one_hot = read_txt_to_dict(MODEL_SAVE_DIR + load_current_model('one-hot-brands'))
    environment = read_config('env')
    print('Current model: ', load_current_model('produce').split('/')[0])
    print('Current env: ', environment)
    app.run(host='0.0.0.0', port=5000)
