from src.manager.model_manager import load_current_model
from src.data_processing.processing import *
from src.model.head import HeadModel
from src.model.lr_model import LRModel
from flask import Flask, jsonify, request
import pandas as pd
from src.config.config import MODEL_SAVE_DIR
from src.utils.util import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

model_produce = LRModel()
model_transition = LRModel()
model_head = HeadModel(STABLE_UNAVAILABLE + TRANSITION_FEATURE_RANGE)
one_hot = None


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
    X_produce, y_produce, delta_produce, mapping_produce = generate_all_training_data(data_per_brand, criterion, one_hot, 'produce')
    metrics_produce = model_produce.train_validate(X_produce, y_produce, delta_produce, mapping_produce)

    # Transition stage
    X_transition, y_transition, delta_transition, mapping_transition = generate_all_training_data(data_per_brand, criterion, one_hot, 'transition')
    metrics_transition = model_transition.train_validate(X_transition, y_transition, delta_transition, mapping_transition)

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


@app.route('/api/test')
def test():
    return 'Ok'


@app.route('/api/predict', methods=["POST"])
def predict():
    data = request.get_json()
    time_ = data['time']
    batch = data['batch']
    index = data['index']
    stage = data['stage']
    brand = data['brand']
    features = data['features']
    if stage == 'produce':
        if len(features) != len(feature_column) * 5 * SPLIT_NUM:
            raise Exception('len(features) wrong, excepted=' + str(len(feature_column) * 5 * SPLIT_NUM) + ' current=' + str(len(features)))
        # features = np.concatenate([features, one_hot[brand]])
        pred = model_produce.predict(features)
    elif stage == 'transition':
        if len(features) != len(feature_column) * 5 * TRANSITION_SPLIT_NUM:
            raise Exception('len(features) wrong, excepted=' + str(len(feature_column) * 5 * SPLIT_NUM) + ' current=' + str(len(features)))
        # features = np.concatenate([features, one_hot[brand]])
        pred = model_transition.predict(features)
    elif stage == 'head':
        pred = model_head.predict(brand, index)
    else:
        raise Exception('param error')

    pred = pred.ravel()

    return jsonify({
        'brand': brand,
        'batch': batch,
        'tempRegion1': pred[0],
        'tempRegion2': pred[1],
        'time': time_,
        'version': '1',
        'deviceStatus': 'deviceStatus'
    })


@app.route('/api/load_model_config')
def api_load_model_config():
    stage = request.args.get("stage")
    if stage == 'produce':
        return jsonify({'window_size': FEATURE_RANGE, 'block_size': int(FEATURE_RANGE / SPLIT_NUM)})
    elif stage == 'transition':
        return jsonify({'window_size': TRANSITION_FEATURE_RANGE, 'block_size': int(TRANSITION_FEATURE_RANGE / TRANSITION_SPLIT_NUM)})
    else:
        raise Exception('param error')


if __name__ == '__main__':
    create_dir(MODEL_SAVE_DIR)
    model_produce.load(MODEL_SAVE_DIR + load_current_model('produce'))
    model_transition.load(MODEL_SAVE_DIR + load_current_model('transition'))
    model_head.load(MODEL_SAVE_DIR + load_current_model('head'))
    one_hot = read_txt_to_dict(MODEL_SAVE_DIR + load_current_model('one-hot-brands'))

    app.run(host='0.0.0.0')
