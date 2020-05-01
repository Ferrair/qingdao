ROOT_PATH = '/Users/qihang/PycharmProjects/qingdao'
MODEL_SAVE_DIR = ROOT_PATH + '/model_save/'
CURRENT_MODE_NAME = None
CONTROL_URL = "http://localhost:64035/api/PLCAPI"


class Environment:
    TEST = 'test'  # 输出到物理测试点位
    PROD = 'prod'  # 输出到生产环境点位
    NONE = 'none'  # 不输出到任何点位
