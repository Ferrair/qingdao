# -*- coding:UTF-8 -*-
import logging

from src.PythonDeviceControlLib.HSControl import *
from src.manager.model_manager import load_current_model
from src.data_processing.processing import *
from src.model.head import HeadModel
from src.model.lr_model import LRModel
from flask import Flask, jsonify, request
import pandas as pd
from src.config.config import MODEL_SAVE_DIR, Environment, ROOT_PATH
from src.utils.util import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

model_produce = LRModel()
model_transition = LRModel()
model_head = HeadModel()
one_hot = None
previous_value = dict()
environment = None
failure = False
brand_strict = True

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


@app.route('/api/predict', methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        keys = data.keys()
        if 'time' not in keys or 'batch' not in keys or 'index' not in keys or 'stage' not in keys or 'brand' not in keys and 'features' not in keys:
            logging.error('Param Error')
            return 'Error'

        storm_time = int(time.time() * 1000)
        time_ = data['time']
        window_time = data.get('window_time', 0)
        model_time = data.get('model_time', 0)
        kafka_time = data.get('kafka_time', 0)
        batch = data['batch']
        index = data['index']
        stage = data['stage']
        brand = data['brand']
        features = data['features']
        originals = []
        device_status = ''
        if 'originals' in keys:
            originals = data['originals']
        if 'device_status' in keys:
            device_status = data['device_status']

        auxiliary_ = get_auxiliary()
    except Exception as e:
        logging.error(e)
        return
    if brand not in one_hot.keys():
        if brand_strict:
            logging.info('our model cannot handle new brand: ' + brand)
            return jsonify('our model cannot handle new brand: ' + brand)
        else:
            brand = 'Txy###'

    if stage == 'produce':
        try:
            check_dim(len(features), len(feature_column) * 5 * SPLIT_NUM)
        except Exception as e:
            logging.error(e)
            return jsonify(str(e))
        else:
            features = np.concatenate([features, auxiliary_, [criterion[brand]], one_hot[brand]])
            pred = model_produce.predict(features)
    elif stage == 'transition':
        try:
            check_dim(len(features), len(feature_column) * 5 * TRANSITION_SPLIT_NUM)
        except Exception as e:
            logging.error(e)
            return jsonify(str(e))
        else:
            features = np.concatenate([features, auxiliary_, [criterion[brand]], one_hot[brand]])
            pred = model_transition.predict(features)
    elif stage == 'head':
        pred = model_head.predict(brand, index)
    else:
        raise Exception('param error')

    pred = pred.ravel()
    pred = adjust(pred, originals)
    pred = clip(pred, temp1_criterion[brand], temp2_criterion[brand])
    pred_time = int(time.time() * 1000)

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
        'time': time_,  # 采样数据的采样时间
        'window_time': window_time,  # 进行Windows划分后的时间
        'model_time': model_time,  # 继续特征计算的时间
        'kafka_time': kafka_time,  # 从Kafka得到数据的时间
        'storm_time': storm_time,  # 数据从Storm过来的时间
        'pred_time': pred_time,  # 预测完成的时间
        'plc_time': int(time.time() * 1000),  # call plc返回后的的时间
        'device_status': device_status,
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

    save_dict_to_txt(ROOT_PATH + '/src/config/env', {
        'env': environment
    })
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
    model_produce.load(MODEL_SAVE_DIR + load_current_model('produce'))
    model_transition.load(MODEL_SAVE_DIR + load_current_model('transition'))
    model_head.load(MODEL_SAVE_DIR + load_current_model('head'))
    one_hot = read_txt_to_dict(MODEL_SAVE_DIR + load_current_model('one-hot-brands'))
    environment = read_txt_to_dict(ROOT_PATH + '/src/config/env')['env']
    print('Current model: ', load_current_model('produce').split('/')[0])
    print('Current env: ', environment)
    app.run(host='0.0.0.0', port=5000)
