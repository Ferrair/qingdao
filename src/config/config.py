ROOT_PATH = '/Users/bytedance/PycharmProjects/qingdao'
MODEL_SAVE_DIR = ROOT_PATH + '/model_save/'
CURRENT_MODE_NAME = None
CONTROL_URL = "http://localhost:64035/api/PLCAPI"

FLOW = '5H.5H.LD5_CK2222_TbcLeafFlowSH'  # '瞬时流量'
BATCH = '6032.6032.LD5_YT603_2B_YS2ROASTBATCHNO'  # '批次'
BRADN = '6032.6032.LD5_YT603_2B_YS2ROASTBRAND'  # '牌号'
HUMID_AFTER_CUT = '6032.6032.LD5_TM2222A_CUTOUTMOISTURE'  # '切丝后出口水分'
TEMP1 = '5H.5H.LD5_KL2226_BucketTemp1SP'  # '一区温度设定值'
TEMP2 = '5H.5H.LD5_KL2226_BucketTemp2SP'  # '二区温度设定值'
HUMID_AFTER_DRYING = '5H.5H.LD5_KL2226_TT1LastMoisPV'  # '烘丝后出口水分'
CUT_HALF_FULL = '6032.6032.LD5_2220_GP2220STATUS3'  # '5000叶丝线暂存柜半满'

WORK_STATUS = '5H.5H.LD5_KL2226_PHASE1'

# TODO：这个2个地方需要修改
# WARM_TEMP1 = '一区预热'
# WARM_TEMP2 = '二区预热'

HUMID_AFTER_CUT_RANGE = 180  # 选取这么多时间的切丝后出口水分平均值
FLOW_LIMIT = 2000  # 流量判断
MIN_DATA_NUM = 2  # 最少的数据限制
HUMID_EPSILON = 0.1  # 低于这个出口水分，几乎就为0，为了防止误差的
HEAD_MAX_TIME = 300  # 头料阶段最大时间，大于这个时间就当做生产状态了


class Environment:
    TEST = 'test'  # 输出到物理测试点位
    PROD = 'prod'  # 输出到生产环境点位
    NONE = 'none'  # 不输出到任何点位
