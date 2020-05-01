import logging

from src.PythonDeviceControlLib.CommandHanlder import CommandHandler
from src.PythonDeviceControlLib.DeviceCommands import DeviceCommandTypes
from src.manager.model_manager import load_current_model
from src.data_processing.processing import *
from src.model.head import HeadModel
from src.model.lr_model import LRModel
from flask import Flask, jsonify, request
import pandas as pd
from src.config.config import MODEL_SAVE_DIR, CONTROL_URL, Environment
from src.utils.util import *
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

model_produce = LRModel()
model_transition = LRModel()
model_head = HeadModel(STABLE_UNAVAILABLE + TRANSITION_FEATURE_RANGE)
one_hot = None
previous_value = dict()
environment = None

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
    model_save_dir = MODEL_SAVE_DIR + get_current_time()

    data = read_data('../data.csv')
    data = data_clean(data)
    data_per_brand, criterion = split_data_by_brand(data)
    one_hot = encode(list(data_per_brand.keys()))

    # Produce stage
    X_produce, y_produce, delta_produce, mapping_produce = generate_all_training_data(data_per_brand,
                                                                                      criterion,
                                                                                      one_hot,
                                                                                      'produce')
    metrics_produce = model_produce.train_validate(X_produce, y_produce, delta_produce, mapping_produce)

    # Transition stage
    X_transition, y_transition, delta_transition, mapping_transition = generate_all_training_data(data_per_brand,
                                                                                                  criterion,
                                                                                                  one_hot,
                                                                                                  'transition')
    metrics_transition = model_transition.train_validate(X_transition,
                                                         y_transition,
                                                         delta_transition,
                                                         mapping_transition)

    # Head stage
    init_per_brand, stable_per_brand = generate_head_dict(data_per_brand, criterion)
    model_head.train(init_per_brand, stable_per_brand)

    # save model
    os.makedirs(model_save_dir)
    model_produce_save_path = model_save_dir + '#produce#' + str(round(metrics_produce['mae'], 3))
    model_transition_save_path = model_save_dir + '#transition#' + str(round(metrics_transition['mae'], 3))
    model_head_save_path = model_save_dir + '#head'
    one_hot_save_path = model_save_dir + '#one-hot-brands'

    model_produce.save(model_produce_save_path)
    model_transition.save(model_transition_save_path)
    model_head.save(model_head_save_path)
    save_dict_to_txt(one_hot_save_path, one_hot)


def validate(data_per_batch: pd.DataFrame) -> np.array:
    test_produce, _ = generate_validation_data(data_per_batch, 'produce')
    test_transition, _ = generate_validation_data(data_per_batch, 'transition')

    pred_produce = model_produce.predict(test_produce)
    pred_transition = model_transition.predict(test_transition)
    pred_head = predict_head(data_per_batch, model_head.init_per_brand, model_head.stable_per_brand)

    pred = np.concatenate([pred_head, pred_transition, pred_produce], axis=0)
    return pred


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
def predict():
    try:
        data = request.get_json()
        keys = data.keys()
        if 'time' not in keys or 'batch' not in keys or 'index' not in keys or 'stage' not in keys or 'brand' not in keys and 'features' not in keys:
            logging.error('Param Error')
            return 'Error'

        storm_time = int(time.time() * 1000)
        time_ = data['time']
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
        logging.info('our model cannot handle new brand: ' + brand)
        return jsonify('our model cannot handle new brand: ' + brand)
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

    if environment == Environment.PROD:
        try:
            roll_back = False
            res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL, [str(pred[0]), str(pred[1])])
            res = json.loads(res.decode())
            logging.info(res)
            logging_in_disk(res)
            for r in res:
                roll_back = roll_back or not r['IsSetSuccessful']
                if r['Address'] == '5H.5H.LD5_KL2226_TT1StandardTemp1':
                    previous_value['T1'] = r['PreviousValue']
                elif r['Address'] == '5H.5H.LD5_KL2226_TT1StandardTemp2':
                    previous_value['T2'] = r['PreviousValue']

            if roll_back:
                res = handler.RunPLCCommand(
                    DeviceCommandTypes.ML_5K_HS_TB_WD_RESET_ALL,
                    [str(previous_value['T1']), str(previous_value['T2'])]
                )
                logging.info(res)
                logging_in_disk(res)
        except Exception as e:
            logging.error(e)
            logging_in_disk(e)
            return 'Error'
    elif environment == Environment.TEST:
        try:
            roll_back = False
            res = handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL, [str(pred[0]), str(pred[1])])
            res = json.loads(res.decode())
            logging.info(res)
            logging_in_disk(res)
            for r in res:
                roll_back = roll_back or not r['IsSetSuccessful']
                if r['Address'] == '5H.5H.LD5_KL2226_TT1StandardTemp1':
                    previous_value['T1'] = r['PreviousValue']
                elif r['Address'] == '5H.5H.LD5_KL2226_TT1StandardTemp2':
                    previous_value['T2'] = r['PreviousValue']

            if roll_back:
                res = handler.RunPLCCommand(
                    DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL,
                    [str(previous_value['T1']), str(previous_value['T2'])]
                )
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
        'time': time_,  # sample time
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
    with open('../logs/pred_log.txt', 'a', buffering=1024 * 10) as f:
        f.write(str(get_current_time()) + ' ---- ' + str(s) + '\n')


def logging_in_disk(s):
    """
    写Logs，所有的Logs都写到Disk里面去了
    """
    with open('../logs/all_log.txt', 'a', buffering=1024 * 10) as f:
        f.write(str(get_current_time()) + ' ---- ' + str(s) + '\n')


@app.route('/api/manual_reset', methods=["POST"])
def manual_reset():
    data = request.get_json()
    T1 = data['T1']
    T2 = data['T2']

    res = handler.RunPLCCommand(
        DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL,
        [str(T1), str(T2)]
    )
    logging.info(res)
    return res


@app.route('/api/manual_reset', methods=["POST"])
def manual_set():
    data = request.get_json()
    T1 = data['T1']
    T2 = data['T2']

    res = handler.RunPLCCommand(
        DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL,
        [str(T1), str(T2)]
    )
    logging.info(res)
    return res


@app.route('/api/get_environment')
def test():
    return jsonify(environment)


@app.route('/api/change_env')
def change_env():
    env = request.args.get("env")
    if env != Environment.NONE and env != Environment.PROD and env != Environment.TEST:
        return jsonify('Error')
    global environment
    environment = env

    save_dict_to_txt('./config/env', {
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
    handler = CommandHandler(CONTROL_URL)
    environment = read_txt_to_dict('./config/env')['env']
    print('Current model: ', load_current_model('produce').split('/')[0])
    print('Current env: ', environment)
    app.run(host='0.0.0.0')

# for test use
#
# X_test = np.array(X_test)
# feature_slice = np.array(np.vsplit(X_test, SPLIT_NUM))
# feature = np.concatenate([
#     np.mean(feature_slice, axis=1).ravel(),
#     np.std(feature_slice, axis=1).ravel(),
#     calc_integral(feature_slice).ravel(),
#     skew(feature_slice, axis=1).ravel(),
#     kurtosis(feature_slice, axis=1).ravel(),
# ])
# feature = feature.ravel()
# feature = np.concatenate([feature, one_hot_dict['TG####A']])
# feature = feature.reshape((1, len(feature)))
# a = scaler.transform(feature)
# a_pred = clf.predict(a)
