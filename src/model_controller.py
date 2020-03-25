import logging

from src.PythonDeviceControlLib.CommandHanlder import CommandHandler
from src.PythonDeviceControlLib.DeviceCommands import DeviceCommandTypes
from src.manager.model_manager import load_current_model
from src.data_processing.processing import *
from src.model.head import HeadModel
from src.model.lr_model import LRModel
from flask import Flask, jsonify, request
import pandas as pd
from src.config.config import MODEL_SAVE_DIR, CONTROL_URL
from src.config import config
from src.utils.util import *
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

model_produce = LRModel()
model_transition = LRModel()
model_head = HeadModel(STABLE_UNAVAILABLE + TRANSITION_FEATURE_RANGE)
one_hot = None
previous_value = dict()

# TODO: not hard code here
criterion = {'Txy###': 12.699999999999994,
             'TG####A': 12.493271237066992,
             'HSX###': 13.80000000000001,
             'TH####A': 12.49285817787605,
             'DQMr##': 13.799999999999997,
             'ThQD##A': 12.5,
             'HsxY##': 13.5,
             'HR####': 12.8}


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


@app.route('/api/test')
def test():
    return config.ENV


# noinspection DuplicatedCode
@app.route('/api/predict', methods=["POST"])
def predict():
    # TODO add logs
    data = request.get_json()
    time_ = data['time']
    batch = data['batch']
    index = data['index']
    stage = data['stage']
    brand = data['brand']
    features = data['features']
    auxiliary_ = get_auxiliary()

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
    if config.ENV == 'prod':
        try:
            roll_back = False
            res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL, [str(pred[0]), str(pred[1])])
            res = json.loads(res.decode())
            logging.info(res)
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
        except Exception as e:
            logging.error(e)
            return e
    else:
        try:
            roll_back = False
            res = handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL, [str(pred[0]), str(pred[1])])
            res = json.loads(res.decode())
            logging.info(res)
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
        except Exception as e:
            logging.error(e)
            return e

    return jsonify({
        'brand': brand,
        'batch': batch,
        'tempRegion1': pred[0],
        'tempRegion2': pred[1],
        'time': time_,  # sample time
        'predTime': int(time.time() * 1000),  # predict time
        'version': '1.2',
        'deviceStatus': 'deviceStatus'
    })


@app.route('/api/reset', methods=["POST"])
def reset():
    data = request.get_json()
    T1 = data['T1']
    T2 = data['T2']

    res = handler.RunPLCCommand(
        DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL,
        [str(T1), str(T2)]
    )
    logging.info(res)
    return res


@app.route('/api/set', methods=["POST"])
def set_():
    data = request.get_json()
    T1 = data['T1']
    T2 = data['T2']

    res = handler.RunPLCCommand(
        DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL,
        [str(T1), str(T2)]
    )
    logging.info(res)
    return res


@app.route('/api/change_env')
def change_env():
    env = request.args.get("env")
    if env in ['prod', 'test']:
        config.ENV = env
        return env
    return 'Fail'


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
    print('Current model: ', load_current_model('produce').split('/')[0])
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
