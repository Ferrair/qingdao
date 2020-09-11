ROOT_PATH = '/Users/bytedance/PycharmProjects/qingdao'
MODEL_SAVE_DIR = ROOT_PATH + '/model_save/'
CURRENT_MODE_NAME = None
CONTROL_URL = "http://localhost:64035/api/PLCAPI"
CONFIG_URL = "http://10.100.100.115:8077/api/ParmStandard/GetTable"

FLOW = '5H.5H.LD5_CK2222_TbcLeafFlowSH'  # '瞬时流量'
BATCH = '6032.6032.LD5_YT603_2B_YS2ROASTBATCHNO'  # '批次'
BRADN = '6032.6032.LD5_YT603_2B_YS2ROASTBRAND'  # '牌号'
HUMID_AFTER_CUT = '6032.6032.LD5_TM2222A_CUTOUTMOISTURE'  # '切丝后出口水分'
TEMP1 = '5H.5H.LD5_KL2226_BucketTemp1SP'  # '一区温度设定值'
TEMP2 = '5H.5H.LD5_KL2226_BucketTemp2SP'  # '二区温度设定值'
TEMP1_CURRENT = '5H.5H.LD5_KL2226_BucketTemp1PV'  # '一区温度实际值'
TEMP2_CURRENT = '5H.5H.LD5_KL2226_BucketTemp2PV'  # '二区温度实际值'
HUMID_AFTER_DRYING = '5H.5H.LD5_KL2226_TT1LastMoisPV'  # '烘丝后出口水分'
HUMID_BEFORE_DRYING = '5H.5H.LD5_KL2226_InputMoisture'  # '烘丝后入口水分'
CUT_HALF_FULL = '6032.6032.LD5_2220_GP2220STATUS3'  # '5000叶丝线暂存柜半满'

TEST_TEMP1 = '5H.5H.LD5_KL2226_BucketTemp1SP'  # '一区温度设定值'
TEST_TEMP2 = '5H.5H.LD5_KL2226_BucketTemp2SP'  # '二区温度设定值'

WORK_STATUS1 = '5H.5H.LD5_KL2226_PHASE1'
WORK_STATUS2 = '5H.5H.LD5_KL2226_PHASE2'

DEFAULT_STANDARD_1 = 135
DEFAULT_STANDARD_2 = 120

MAX_BEFORE_HUMID_SIZE = 20  # 利用多长时间的入口水分
HUMID_AFTER_CUT_RANGE = 180  # 选取这么多时间的切丝后出口水分平均值
FLOW_LIMIT = 2000  # 流量判断
FLOW_MIN = 10  # 流量判断
MIN_DATA_NUM = 2  # 最少的数据限制
HUMID_EPSILON = 0.1  # 低于这个出口水分，几乎就为0，为了防止误差的
HEAD_MAX_TIME = 300  # 头料阶段最大时间，大于这个时间就当做生产状态了

# 烘干后出口水分设定值
criterion = {'Txy###': 12.699999999999994,
             'TG####A': 12.493271237066992,
             'HSX###': 13.80000000000001,
             'TH####A': 12.49285817787605,
             'DQMr##': 13.799999999999997,
             'ThQD##A': 12.5,
             'HsxY##': 13.5,
             'HR####': 12.8}


class Environment:
    TEST = 'test'  # 输出到物理测试点位
    PROD = 'prod'  # 输出到生产环境点位
    NONE = 'none'  # 不输出到任何点位
