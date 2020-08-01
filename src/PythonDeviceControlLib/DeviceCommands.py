# -*- coding:UTF-8 -*-
from enum import Enum

from src.PythonDeviceControlLib.DataTypes import PLCComand, PLCComandList
from src.PythonDeviceControlLib.PLog import logger


class DeviceCommandTypes(Enum):
    ML_5K_HS_TB_WD_SET_ALL = 0  # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置五千线烘丝机筒壁1区、2区温度设定值

    ML_5K_HS_TB_WD_RESET_ALL = 1  # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:恢复五千线烘丝机筒壁1区、2区温度设定值

    ML_5K_HS_TB_WD_READ_HMI = 2  # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置五千线烘丝机筒壁1区、2区温度,读取HMI判断

    ML_5K_HS_TB_WD_READ_TEMPS = 3  # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:读取当前五千线烘丝机筒壁1区、2区温度

    ML_5K_HS_TB_WD_SET_TEMPS = 4  # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置当前五千线烘丝机筒壁1区、2区温度,不执行切换

    ML_5K_HS_TB_WD_SET_TIC_COS = 5  # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置当前五千线烘丝机筒壁1区、2区正向控制除水

    ML_5K_HS_TB_WD_SWITCH_MAN = 6  # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置五千线烘丝机筒壁1区、2区温度,切换到Hauni手工

    ML_5K_HS_TB_WD_SWITCH_AUTO = 7  # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置五千线烘丝机筒壁1区、2区温度,切换到Hauni自动

    SIM_TEST_D1_T1 = 8  # test:SIM_TEST_D1_T1

    SIM_TEST_D1_T2 = 9  # test:SIM_TEST_D1_T2

    SIM_TEST_D1_T4 = 10  # test:SIM_TEST_D1_T4

    ML_5H_5H_LD5_TEST_SET_ALL = 11  # test:ML_5H_5H_LD5_TEST_SET_ALL

    ML_5H_5H_LD5_TEST_RESET_ALL = 12  # test:ML_5H_5H_LD5_TEST_RESET_ALL

    ZZ_5K_HS_YYHC_SET_YYHCLL = 13  # 3.1.	烟叶回潮:3.1.1.	设置烟叶回潮流量

    ZZ_5K_HS_YYHC_SET_SFYTDH = 14  # 3.1.	烟叶回潮:3.1.2.	设置水分仪通道号

    ZZ_5K_HS_YYHC_SET_RFWD = 15  # 3.1.	烟叶回潮:3.1.3.	设置热风温度

    ZZ_5K_HS_YYHC_SET_RKSF = 16  # 3.1.	烟叶回潮:3.1.4.	设置入口水分

    ZZ_5K_HS_YYHC_SET_CKSF = 17  # 3.1.	烟叶回潮:3.1.5.	设置出口水分

    ZZ_5K_HS_YYHC_SET_JSXS = 18  # 3.1.	烟叶回潮:3.1.6.	设置加水系数

    ZZ_5K_HS_YYHC_SET_ZKDZCLL = 19  # 3.1.	烟叶回潮:3.1.7.	设置真空电子秤流量

    ZZ_5K_HS_BPHC_SET_BPHCLL = 20  # 3.2.	薄片回潮:3.2.1.	设置薄片回潮流量

    ZZ_5K_HS_BPHC_SET_SFYTDH = 21  # 3.2.	薄片回潮:3.2.2.	设置水分仪通道号

    ZZ_5K_HS_BPHC_SET_RFWD = 22  # 3.2.	薄片回潮:3.2.3.	设置热风温度

    ZZ_5K_HS_BPHC_SET_RKSF = 23  # 3.2.	薄片回潮:3.2.4.	设置入口水分

    ZZ_5K_HS_BPHC_SET_CKSF = 24  # 3.2.	薄片回潮:3.2.5.	设置出口水分

    ZZ_5K_HS_BPHC_SET_JSBL = 25  # 3.2.	薄片回潮:3.2.6.	设置加水比例

    ZZ_5K_HS_BPHC_SET_CKSFSX = 26  # 3.2.	薄片回潮:3.2.7.	设置出口水分上限设定值

    ZZ_5K_HS_BPHC_SET_CKSFXX = 27  # 3.2.	薄片回潮:3.2.8.	设置出口水分下限设定值

    ZZ_5K_HS_BPHC_SET_CKSFBL = 28  # 3.2.	薄片回潮:3.2.9.	设置入口水分比例设定值

    ZZ_5K_HS_BPHC_SET_TLJSL = 29  # 3.2.	薄片回潮:3.2.10.	头料加水量

    ZZ_5K_HS_BPHC_SET_WLJSL = 30  # 3.2.	薄片回潮:3.2.11.	设置尾料加水量

    ZZ_5K_HS_JL_SET_YYLL = 31  # 3.3.	加料:3.3.1.	设置烟叶流量

    ZZ_5K_HS_JL_SET_SFYTDH = 32  # 3.3.	加料:3.3.2.	设置水分仪通道号

    ZZ_5K_HS_JL_SET_JLBL = 33  # 3.3.	加料:3.3.3.	设置加料比例

    ZZ_5K_HS_JL_SET_RFWD = 34  # 3.3.	加料:3.3.4.	设置热风温度

    ZZ_5K_HS_JL_SET_LT1WD = 35  # 3.3.	加料:3.3.5.	设置料桶1温度

    ZZ_5K_HS_JL_SET_LT2WD = 36  # 3.3.	加料:3.3.6.	设置料桶2温度

    ZZ_5K_HS_JL_SET_TLXZXS = 37  # 3.3.	加料:3.3.7.	设置头料修正系数

    ZZ_5K_HS_JL_SET_WLXZXS = 38  # 3.3.	加料:3.3.8.	设置尾料修正系数

    ZZ_5K_HS_JL_SET_WLLL = 39  # 3.3.	加料:3.3.9.	设置尾料流量

    ZZ_5K_HS_JL_SET_JSXS = 40  # 3.3.	加料:3.3.10.	设置加水系数

    ZZ_5K_HS_JL_SET_JSBL = 41  # 3.3.	加料:3.3.11.	设置加水比例

    ZZ_5K_HS_JL_SET_CKHSL = 42  # 3.3.	加料:3.3.12.	设置出口含水量

    ZZ_5K_HS_JL_SET_DSFW = 43  # 3.3.	加料:3.3.13.	设置带速范围

    ZZ_5K_HS_JL_READ_YLAN = 44  # 3.3.	加料:3.3.14.	设置要料重量

    ZZ_5K_HS_JL_SET_LTXZ1 = 45  # 3.3.	加料:3.3.14.	设置要料重量

    ZZ_5K_HS_JL_SET_LTXZ2 = 46  # 3.3.	加料:3.3.14.	设置要料重量

    ZZ_5K_HS_JL_READ_LYAN = 47  # 3.3.	加料:3.3.15.	设置料液代码

    ZZ_5K_HS_JL_SET_LYDM1 = 48  # 3.3.	加料:3.3.15.	设置料液代码

    ZZ_5K_HS_JL_SET_LYDM2 = 49  # 3.3.	加料:3.3.15.	设置料液代码

    ZZ_5K_HS_HS_SET_LLLL = 50  # 3.4.	烘丝 :3.4.1.	设置来料流量

    ZZ_5K_HS_HS_SET_SIROX = 51  # 3.4.	烘丝 :3.4.3.	设置SIROX蒸汽流量

    ZZ_5K_HS_HS_SET_RFFS = 52  # 3.4.	烘丝 :3.4.3.	设置热风风速

    ZZ_5K_HS_HS_SET_RFWD = 53  # 3.4.	烘丝 :3.4.4.	设置热风温度

    ZZ_5K_HS_HS_SET_CKSF = 54  # 3.4.	烘丝 :3.4.5.	设置出口水分

    ZZ_5K_HS_HS_SET_SFYTD = 55  # 3.4.	烘丝 :3.4.6.	设置水分仪通道

    ZZ_5K_HS_HS_SET_TSSD = 56  # 3.4.	烘丝 :3.4.7.	设置脱水速度

    ZZ_5K_HS_HCJLHS_SET_HCZHQD = 57  # 3.5.	回潮/加料/烘丝:3.5.1.	设置回潮组合启动

    ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT1XZ = 58  # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动:料筒1选择

    ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT1ZL = 59  # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动:料筒1重量

    ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT2XZ = 60  # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动:料筒2选择

    ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT2ZL = 61  # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动:料筒2重量

    ZZ_5K_HS_HCJLHS_SET_JLZHQD = 62  # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动

    ZZ_5K_HS_HCJLHS_SET_QSZHQD = 63  # 3.5.	回潮/加料/烘丝:3.5.1.	设置切丝组合启动

    ZZ_5K_HS_HCJLHS_SET_HSZHQD = 64  # 3.5.	回潮/加料/烘丝:3.5.1.	设置烘丝组合启动

    ZZ_5K_HS_HCJLHS_SET_HCZHTZ = 65  # 3.5.	回潮/加料/烘丝:3.5.2.	设置组合停止：回潮组合停止

    ZZ_5K_HS_HCJLHS_SET_JLZHTZ = 66  # 3.5.	回潮/加料/烘丝:3.5.2.	设置组合停止：加料组合停止

    ZZ_5K_HS_HCJLHS_SET_QSZHTZ = 67  # 3.5.	回潮/加料/烘丝:3.5.2.	设置组合停止：切丝段组合停止

    ZZ_5K_HS_HCJLHS_SET_HSZHTZ = 68  # 3.5.	回潮/加料/烘丝:3.5.2.	设置组合停止：烘丝段组合停止

    ZZ_5K_HS_HCJLHS_SET_SZYR_HCYRQD = 69  # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:回潮预热启动

    ZZ_5K_HS_HCJLHS_SET_SZYR_HCYRTZ = 70  # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:回潮预热停止

    ZZ_5K_HS_HCJLHS_SET_SZYR_JLYRQD = 71  # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:加料预热启动

    ZZ_5K_HS_HCJLHS_SET_SZYR_JLYRTZ = 72  # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:加料预热停止

    ZZ_5K_HS_HCJLHS_READ_KZMS = 73  # 3.5.	回潮/加料/烘丝:3.5.3.	5000烘丝机控制模式判断

    ZZ_5K_HS_HCJLHS_SET_SZYR_SIROX_YRQD = 74  # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:SIROX预热启动

    ZZ_5K_HS_HCJLHS_SET_SZYR_SIROX_YRTZ = 75  # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:SIROX预热停止

    ZZ_5K_HS_HCJLHS_SET_SZYR_HSYRQD = 76  # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:烘丝预热启动

    ZZ_5K_HS_HCJLHS_SET_SZYR_HSYRTZ = 77  # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:烘丝预热停止

    ZZ_5K_HS_HCJLHS_SET_SZLQ_HSLQKS = 78  # 3.5.	回潮/加料/烘丝:3.5.4.	设置冷却:烘丝冷却开始

    ZZ_5K_HS_HCJLHS_SET_SZLQ_HSLQTZ = 79  # 3.5.	回潮/加料/烘丝:3.5.4.	设置冷却:烘丝冷却停止

    ZZ_5K_HS_HCJLHS_SET_SZLQ_SIROXLQKS = 80  # 3.5.	回潮/加料/烘丝:3.5.4.	设置冷却:SIROX冷却开始

    ZZ_5K_HS_HCJLHS_SET_SZLQ_SIROXLQTZ = 81  # 3.5.	回潮/加料/烘丝:3.5.4.	设置冷却:SIROX冷却停止

    ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCKS = 82  # 3.5.	回潮/加料/烘丝:3.5.5.	设置生产停止:烘丝机生产开始

    ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCTZ = 83  # 3.5.	回潮/加料/烘丝:3.5.5.	设置生产停止:烘丝机生产停止

    ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCZT = 84  # 3.5.	回潮/加料/烘丝:3.5.5.	设置生产停止:烘丝机生产暂停

    ZZ_5K_HS_HCJLHS_SET_SZFW = 85  # 3.5.	回潮/加料/烘丝:3.5.6.	设置复位

    ZZ_5K_HS_HCJLHS_SET_SZQX_SIROXQJKS = 86  # 3.5.	回潮/加料/烘丝:3.5.7.	设置清洗:SIROX清洁开始

    ZZ_5K_HS_HCJLHS_SET_SZQX_SIROXQJTZ = 87  # 3.5.	回潮/加料/烘丝:3.5.7.	设置清洗:SIROX清洁停止

    ML_5K_HS_HC_SET_JSXS = 88  # 2.4.	回潮模型加水系数:智能模型输出加水系数

    ML_5K_HS_JL_SET_JSXS = 89  # 2.5.	加料模型加水系数:智能模型输出加料模型加水系数

    ML_5K_HS_JL_SET_JSBL = 90  # 2.5.	加料模型加水比例:智能模型输出加料模型加水比例

    ZZ_5K_HS_YYHC_SET_YPGXGDZD = 91  # 3.1.	烟叶回潮:3.1.8.	设置预配柜选柜打自动

    ZZ_5K_HS_YYHC_SET_YPGLJXZ = 92  # 3.1.	烟叶回潮:3.1.8.	设置预配柜选柜打自动--路径选择

    ZZ_5K_HS_JL_SET_YPGXGDZD = 93  # 3.3.	加料:3.3.16.	设置预配柜选柜打自动

    ZZ_5K_HS_JL_SET_YGRGZTZD = 94  # 3.3.	加料:3.3.18.	设置叶柜入柜状态自动

    ZZ_5K_HS_JL_SET_YGXZLJ = 95  # 3.3.	加料:3.3.19.	设置叶柜选择路径

    ZZ_5K_HS_HCJLHS_SET_SCTS_JLSCTS = 96  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-加料生产提升

    ZZ_5K_HS_HCJLHS_SET_SCTS_JLQZTZ = 97  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-加料强制停止

    ZZ_5K_HS_HCJLHS_SET_SCTS_JLQZTS = 98  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-加料强制提升

    ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCSCTS = 99  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-薄片回潮生产提升

    ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCQZTZ = 100  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-薄片回潮强制停止

    ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCQZTS = 101  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-薄片回潮强制提升

    ZZ_5K_HS_HCJLHS_SET_SCTS_HCSCTS = 102  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-回潮生产提升

    ZZ_5K_HS_HCJLHS_SET_SCTS_HCQZTZ = 103  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-回潮强制停止

    ZZ_5K_HS_HCJLHS_SET_SCTS_HCQZTS = 104  # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-回潮强制提升


class DeviceCommandGenerator:

    def __init__(self):

        # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置五千线烘丝机筒壁1区、2区温度设定值
        self.Set5KTempAll = [
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_1", "float", "0", "", "", ""),  # 正向除水1
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_2", "float", "0", "", "", ""),  # 正向除水2
            PLCComand("5H.5H.LD5_KL2226_PID04_CV", "float", "0", "", "", ""),  # 关闭反馈
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp1", "float", "{0}", "", "", ""),  # 1区温度
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp2", "float", "{1}", "", "", "")  # 2区温度
        ]

        # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:恢复五千线烘丝机筒壁1区、2区温度设定值
        self.ReSet5KTempAll = [
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_2", "float", "1", "", "", ""),  # 正向除水2
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_1", "float", "1", "", "", ""),  # 正向除水1
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp1", "float", "{0}", "", "", ""),  # 1区温度
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp2", "float", "{1}", "", "", "")  # 2区温度
        ]

        # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置五千线烘丝机筒壁1区、2区温度,读取HMI判断
        self.Read5KTBWDTempHMI = [
            PLCComand("", "", "", "5H.5H.LD5_KL2226_PID042MCVHMI", "boolean", "")  # 读取HMI判断Hauni状态
        ]

        # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:读取当前五千线烘丝机筒壁1区、2区温度
        self.Read5KTBWDTemps = [
            PLCComand("", "", "", "5H.5H.LD5_KL2226_TT1StandardTemp1", "float", ""),  # 读取1区温度
            PLCComand("", "", "", "5H.5H.LD5_KL2226_TT1StandardTemp2", "float", "")  # 读取2区温度
        ]

        # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置当前五千线烘丝机筒壁1区、2区温度,不执行切换
        self.Set5KTBWDTemps = [
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp1", "float", "{0}", "", "", ""),  # 设置1区温度
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp2", "float", "{1}", "", "", "")  # 设置2区温度
        ]

        # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置当前五千线烘丝机筒壁1区、2区正向控制除水
        self.Set5KTBWDTTICCOS = [
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_1", "float", "{0}", "", "", ""),  # 设置1区正向控制除水
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_2", "float", "{1}", "", "", "")  # 设置2区正向控制除水
        ]

        # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置五千线烘丝机筒壁1区、2区温度,切换到Hauni手工
        self.Set5KTBWDSwitchMan = [
            PLCComand("5H.5H.LD5_KL2226_PID04_OPERATE1_MAN", "boolean", "True", "5H.5H.LD5_KL2226_PID042MCVHMI",
                      "boolean", "True")  # 切换到Hauni手工模式
        ]

        # 2.1.	五千线烘丝机筒壁1区、2区温度控制流程:设置五千线烘丝机筒壁1区、2区温度,切换到Hauni自动
        self.Set5KTBWDSwitchAuto = [
            PLCComand("5H.5H.LD5_KL2226_PID04OPERATE1AUTO", "boolean", "True", "5H.5H.LD5_KL2226_PID042MCVHMI",
                      "boolean", "False")  # 切换到Hauni自动模式
        ]

        # test:SIM_TEST_D1_T1
        self.SetSIMTestD1T1 = [
            PLCComand("test.d1.t1", "float", "{0}", "", "", "")  # 测试点位1,ML
        ]

        # test:SIM_TEST_D1_T2
        self.SetSIMTestD1T2 = [
            PLCComand("test.d1.t2", "float", "{0}", "", "", "")  # 测试点位2,ML
        ]

        # test:SIM_TEST_D1_T4
        self.SetSIMTestD1T4 = [
            PLCComand("test.d1.t4", "float", "{0}", "", "", "")  # 测试点位3,APP
        ]

        # test:ML_5H_5H_LD5_TEST_SET_ALL
        self.SetLD5Test = [
            PLCComand("5H.5H.LD5_test1", "float", "{0}", "", "", ""),  # 设置五千线烘丝机筒壁1区、2区温度测试值设定test1
            PLCComand("5H.5H.LD5_test2", "float", "{1}", "", "", "")  # 设置五千线烘丝机筒壁1区、2区温度测试值设定test2
        ]

        # test:ML_5H_5H_LD5_TEST_RESET_ALL
        self.ReSetLD5Test = [
            PLCComand("5H.5H.LD5_test1", "float", "{0}", "", "", ""),  # 恢复五千线烘丝机筒壁1区、2区温度测试值恢复test1
            PLCComand("5H.5H.LD5_test2", "float", "{1}", "", "", "")  # 恢复五千线烘丝机筒壁1区、2区温度测试值恢复test2
        ]

        # 3.1.	烟叶回潮:3.1.1.	设置烟叶回潮流量
        self.Set5KYYHCLL = [
            PLCComand("6022.6022.LAP5_DP2113_Flow_SET", "float", "{0}", "", "", "")  # 烟叶回潮流量
        ]

        # 3.1.	烟叶回潮:3.1.2.	设置水分仪通道号
        self.Set5KYYHCSFYTDH = [
            PLCComand("6022.6022.LAP5_3375_ROAD_S", "int", "{0}", "", "", ""),  # 水分仪通道号3375
            PLCComand("6022.6022.LAP5_2118_ROAD_S", "int", "{1}", "", "", "")  # 水分仪通道号2118
        ]

        # 3.1.	烟叶回潮:3.1.3.	设置热风温度
        self.Set5KYYHCRFWD = [
            PLCComand("6022.6022.LAP5_WQ2116_HotWind_SP", "float", "{0}", "", "", "")  # 热风温度
        ]

        # 3.1.	烟叶回潮:3.1.4.	设置入口水分
        self.Set5KYYHCRKSF = [
            PLCComand("6022.6022.LAP5_WQ2116_InMois_SET", "float", "{0}", "", "", "")  # 入口水分
        ]

        # 3.1.	烟叶回潮:3.1.5.	设置出口水分
        self.Set5KYYHCCKSF = [
            PLCComand("6022.6022.LAP5_WQ2116_OutMois_SP", "float", "{0}", "", "", "")  # 出口水分
        ]

        # 3.1.	烟叶回潮:3.1.6.	设置加水系数
        self.Set5KYYHCJSXS = [
            PLCComand("6022.6022.LAP5_WQ2116_AddWaterRatio1_SET", "float", "{0}", "", "", "")  # 加水系数
        ]

        # 3.1.	烟叶回潮:3.1.7.	设置真空电子秤流量
        self.Set5KYYHCZKDZCLL = [
            PLCComand("6022.6022.LVR_3374_Flow_SET", "float", "{0}", "", "", "")  # 真空电子秤流量
        ]

        # 3.2.	薄片回潮:3.2.1.	设置薄片回潮流量
        self.Set5KBPHCBPHCLL = [
            PLCComand("6022.6022.LAP5_2123_Flow_SET", "float", "{0}", "", "", "")  # 薄片回潮流量
        ]

        # 3.2.	薄片回潮:3.2.2.	设置水分仪通道号
        self.Set5KBPHCSFYTDH = [
            PLCComand("6022.6022.LAP5_2124_ROAD_S", "int", "{0}", "", "", ""),  # 水分仪通道号2124
            PLCComand("6022.6022.LAP5_2129_ROAD_S", "int", "{1}", "", "", "")  # 水分仪通道号2129
        ]

        # 3.2.	薄片回潮:3.2.3.	设置热风温度
        self.Set5KBPHCRFWD = [
            PLCComand("6022.6022.LAP5_WQ2126_HotWind_SP", "float", "{0}", "", "", "")  # 热风温度
        ]

        # 3.2.	薄片回潮:3.2.4.	设置入口水分
        self.Set5KBPHCRKSF = [
            PLCComand("6022.6022.LAP5_WQ2126_InMois_SET", "float", "{0}", "", "", "")  # 入口水分
        ]

        # 3.2.	薄片回潮:3.2.5.	设置出口水分
        self.Set5KBPHCCKSF = [
            PLCComand("6022.6022.LAP5_WQ2126_OutMois_SP", "float", "{0}", "", "", "")  # 出口水分
        ]

        # 3.2.	薄片回潮:3.2.6.	设置加水比例
        self.Set5KBPHCJSBL = [
            PLCComand("6022.6022.LAP5_WQ2126_AddWaterRatio1_SET", "float", "{0}", "", "", "")  # 加水比例
        ]

        # 3.2.	薄片回潮:3.2.7.	设置出口水分上限设定值
        self.Set5KBPHCCKSFSX = [
            PLCComand("6022.6022.LAP5_WQ2126_OutMois_SET_H", "float", "{0}", "", "", "")  # 出口水分上限设定值
        ]

        # 3.2.	薄片回潮:3.2.8.	设置出口水分下限设定值
        self.Set5KBPHCCKSFXX = [
            PLCComand("6022.6022.LAP5_WQ2126_OutMois_SET_L", "float", "{0}", "", "", "")  # 出口水分下限设定值
        ]

        # 3.2.	薄片回潮:3.2.9.	设置入口水分比例设定值
        self.Set5KBPHCCKSFBL = [
            PLCComand("6022.6022.LAP5_WQ2126_AddWaterRatio2_SET", "float", "{0}", "", "", "")  # 入口水分比例设定值
        ]

        # 3.2.	薄片回潮:3.2.10.	头料加水量
        self.Set5KBPHCTLJSL = [
            PLCComand("6022.6022.LAP5_WQ2126_TAddWater_SET", "float", "{0}", "", "", "")  # 头料加水量
        ]

        # 3.2.	薄片回潮:3.2.11.	设置尾料加水量
        self.Set5KBPHCWLJSL = [
            PLCComand("6022.6022.LAP5_WQ2126_HAddWater_SET", "float", "{0}", "", "", "")  # 尾料加水量
        ]

        # 3.3.	加料:3.3.1.	设置烟叶流量
        self.Set5KJLYYLL = [
            PLCComand("6022.6022.LAC5_2143_Flow_SET", "float", "{0}", "", "", "")  # 烟叶流量
        ]

        # 3.3.	加料:3.3.2.	设置水分仪通道号
        self.Set5KJLSFYTDH = [
            PLCComand("6022.6022.LAC5_2143A_ROAD_S", "int", "{0}", "", "", ""),  # 水分仪通道号2143A
            PLCComand("6022.6022.LAC5_2151_ROAD_S", "int", "{1}", "", "", "")  # 水分仪通道号2151
        ]

        # 3.3.	加料:3.3.3.	设置加料比例
        self.Set5KJLJLBL = [
            PLCComand("6022.6022.LAC5_SJ2147_AddFeedRatio_SET", "float", "{0}", "", "", "")  # 加料比例
        ]

        # 3.3.	加料:3.3.4.	设置热风温度
        self.Set5KJLRFWD = [
            PLCComand("6022.6022.LAC5_SJ2147_HotWind_SP", "float", "{0}", "", "", "")  # 热风温度
        ]

        # 3.3.	加料:3.3.5.	设置料桶1温度
        self.Set5KJLLT1WD = [
            PLCComand("6022.6022.LAC5_SJ2147_TankTemp_SET", "float", "{0}", "", "", "")  # 料桶1温度
        ]

        # 3.3.	加料:3.3.6.	设置料桶2温度
        self.Set5KJLLT2WD = [
            PLCComand("6022.6022.LAC5_SJ2147_TankTemp2_SET", "float", "{0}", "", "", "")  # 料桶2温度
        ]

        # 3.3.	加料:3.3.7.	设置头料修正系数
        self.Set5KJLTLXZXS = [
            PLCComand("6022.6022.LAC5_SJ2147_Tcorrection_R_SET", "float", "{0}", "", "", "")  # 头料修正系数
        ]

        # 3.3.	加料:3.3.8.	设置尾料修正系数
        self.Set5KJLWLXZXS = [
            PLCComand("6022.6022.LAC5_SJ2147_Hcorrection_R_SET", "float", "{0}", "", "", "")  # 尾料修正系数
        ]

        # 3.3.	加料:3.3.9.	设置尾料流量
        self.Set5KJLWLLL = [
            PLCComand("6022.6022.LAC5_SJ2147_Tflow", "float", "{0}", "", "", "")  # 尾料流量
        ]

        # 3.3.	加料:3.3.10.	设置加水系数
        self.Set5KJLJSXS = [
            PLCComand("6022.6022.LAC5_SJ2147_AddWaterRatio1_SET", "float", "{0}", "", "", "")  # 加水系数
        ]

        # 3.3.	加料:3.3.11.	设置加水比例
        self.Set5KJLJSBL = [
            PLCComand("6022.6022.LAC5_SJ2147_AddWaterRatio2_SET", "float", "{0}", "", "", "")  # 加水比例
        ]

        # 3.3.	加料:3.3.12.	设置出口含水量
        self.Set5KJLCKHSL = [
            PLCComand("6022.6022.LAC5_SJ2147_OutMois_SP", "float", "{0}", "", "", "")  # 出口含水量
        ]

        # 3.3.	加料:3.3.13.	设置带速范围
        self.Set5KJLDSFW = [
            PLCComand("", "", "", "6022.6022.LAC5_2143_Speed", "float", "")  # 带速范围（实际值,只读）
        ]

        # 3.3.	加料:3.3.14.	设置要料重量
        self.Read5KJLYLAN = [
            PLCComand("", "", "", "6022.6022.LAC5_SJ2147_TANK1_REQUEST", "boolean", ""),  # 要料按钮1
            PLCComand("", "", "", "6022.6022.LAC5_SJ2147_TANK2_REQUEST", "boolean", "")  # 要料按钮2
        ]

        # 3.3.	加料:3.3.14.	设置要料重量
        self.Set5KJLLTXZ1 = [
            PLCComand("6022.6022.LAC5_SJ2147_TANK1_SEL", "boolean", "True", "", "", ""),  # 料筒1选择
            PLCComand("LF.LF.LAC5_SJ2147_TANK1ASKQTY", "string", "{0}", "", "", "")  # 料筒1要料重量
        ]

        # 3.3.	加料:3.3.14.	设置要料重量
        self.Set5KJLLTXZ2 = [
            PLCComand("6022.6022.LAC5_SJ2147_TANK2_SEL", "boolean", "True", "", "", ""),  # 料筒2选择
            PLCComand("LF.LF.LAC5_SJ2147_TANK2ASKQTY", "string", "{0}", "", "", "")  # 料筒2要料重量
        ]

        # 3.3.	加料:3.3.15.	设置料液代码
        self.Read5KJLLYAN = [
            PLCComand("", "", "", "6022.6022.LAC5_SJ2147_TANK1_REQUEST", "boolean", ""),  # 要料按钮1
            PLCComand("", "", "", "6022.6022.LAC5_SJ2147_TANK2_REQUEST", "boolean", "")  # 要料按钮2
        ]

        # 3.3.	加料:3.3.15.	设置料液代码
        self.Set5KJLLYDM1 = [
            PLCComand("6022.6022.LAC5_SJ2147_TANK1_SEL", "boolean", "True", "", "", ""),  # 料筒1选择
            PLCComand("LF.LF.LAC5_SJ2147_TANK1ASKBRAND", "string", "{0}", "", "", "")  # 料筒1料液代码
        ]

        # 3.3.	加料:3.3.15.	设置料液代码
        self.Set5KJLLYDM2 = [
            PLCComand("6022.6022.LAC5_SJ2147_TANK2_SEL", "boolean", "True", "", "", ""),  # 料筒2选择
            PLCComand("LF.LF.LAC5_SJ2147_TANK2ASKBRAND", "string", "{0}", "", "", "")  # 料筒2料液代码
        ]

        # 3.4.	烘丝 :3.4.1.	设置来料流量
        self.Set5KHSHSLLLL = [
            PLCComand("5H.5H.LD5_KL2226_TobFlowSP", "float", "{0}", "", "", "")  # 来料流量
        ]

        # 3.4.	烘丝 :3.4.3.	设置SIROX蒸汽流量
        self.Set5KHSHSSIROX = [
            PLCComand("5H.5H.LD5_SX2224_ProcWindFlowSP", "float", "{0}", "", "", "")  # SIROX蒸汽流量
        ]

        # 3.4.	烘丝 :3.4.3.	设置热风风速
        self.Set5KHSHSRFFS = [
            PLCComand("5H.5H.LD5_KL2226_ProcAirSpeedSP", "float", "{0}", "", "", "")  # 热风风速
        ]

        # 3.4.	烘丝 :3.4.4.	设置热风温度
        self.Set5KHSHSRFWD = [
            PLCComand("5H.5H.LD5_KL2226_TT1V31SPProcAirTemp", "float", "{0}", "", "", "")  # 热风温度
        ]

        # 3.4.	烘丝 :3.4.5.	设置出口水分
        self.Set5KHSHSCKSF = [
            PLCComand("5H.5H.LD5_KL2226_PID04SP", "float", "{0}", "", "", "")  # 出口水分
        ]

        # 3.4.	烘丝 :3.4.6.	设置水分仪通道
        self.Set5KHSHSSFYTD = [
            PLCComand("5H.5H.LD5_KL2226_SetpointChannal1", "int", "{0}", "", "", ""),  # 通道1
            PLCComand("5H.5H.LD5_KL2226_SetpointChannal2", "int", "{1}", "", "", "")  # 通道2
        ]

        # 3.4.	烘丝 :3.4.7.	设置脱水速度
        self.Set5KHSHSSTSSD = [
            PLCComand("5H.5H.LD5_KL2226_WatOffDryerSP", "float", "{0}", "", "", "")  # 脱水速度
        ]

        # 3.5.	回潮/加料/烘丝:3.5.1.	设置回潮组合启动
        self.Set5KHCJLHSHCZHQD = [
            PLCComand("6022.6022.LAP5_YT602_2A_TASK1_START", "boolean", "True", "", "", "")  # 回潮组合启动
        ]

        # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动:料筒1选择
        self.Read5KHCJLHSJLZHQDLT1XZ = [
            PLCComand("", "", "", "6022.6022.LAC5_SJ2147_TANK1_SEL", "boolean", "")  # 料筒1选择
        ]

        # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动:料筒1重量
        self.Read5KHCJLHSJLZHQDLT1ZL = [
            PLCComand("", "", "", "6022.6022.LAC5_SJ2147_TankWeight1", "float", "")  # 料筒1重量
        ]

        # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动:料筒2选择
        self.Read5KHCJLHSJLZHQDLT2XZ = [
            PLCComand("", "", "", "6022.6022.LAC5_SJ2147_TANK2_SEL", "boolean", "")  # 料筒2选择
        ]

        # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动:料筒2重量
        self.Read5KHCJLHSJLZHQDLT2ZL = [
            PLCComand("", "", "", "6022.6022.LAC5_SJ2147_TankWeight1", "float", "")  # 料筒2重量
        ]

        # 3.5.	回潮/加料/烘丝:3.5.1.	设置加料组合启动
        self.Set5KHCJLHSJLZHQD = [
            PLCComand("6022.6022.LAC5_YT602_2B_TASK2_START", "boolean", "True", "", "", "")  # 加料组合启动
        ]

        # 3.5.	回潮/加料/烘丝:3.5.1.	设置切丝组合启动
        self.Set5KHCJLHSQSZHQD = [
            PLCComand("6032.6032.LC5_YT603_2A_TASK1START", "boolean", "True", "", "", ""),  # 切丝段启动
            PLCComand("YG.YG.YG_YT630_OUTTASK2START", "boolean", "True", "", "", ""),  # 叶柜出柜任务2启动
            PLCComand("YG.YG.YG_YT630_OUTTASK2STOP", "boolean", "True", "", "", "")  # 叶柜出柜任务2结束
        ]

        # 3.5.	回潮/加料/烘丝:3.5.1.	设置烘丝组合启动
        self.Set5KHCJLHSHSZHQD = [
            PLCComand("6032.6032.LD5_YT603_2B_TASK2START", "boolean", "True", "", "", ""),  # 烘丝段启动
            PLCComand("YSG.YSG.YSG_YT632_INTASK2START", "boolean", "True", "", "", "")  # 叶丝柜进柜任务2启动
        ]

        # 3.5.	回潮/加料/烘丝:3.5.2.	设置组合停止：回潮组合停止
        self.Set5KHCJLHSHCZHTZ = [
            PLCComand("6022.6022.LAP5_YT602_2A_IN_TASK1_STOP", "boolean", "True", "", "", "")  # 回潮组合停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.2.	设置组合停止：加料组合停止
        self.Set5KHCJLHSJLZHTZ = [
            PLCComand("6022.6022.LAC5_YT602_2B_TASK2_STOP", "boolean", "True", "", "", ""),  # 加料段结束
            PLCComand("YG.YG.YG_YT630_INTASK2STOP", "boolean", "True", "", "", "")  # 叶柜进柜段结束
        ]

        # 3.5.	回潮/加料/烘丝:3.5.2.	设置组合停止：切丝段组合停止
        self.Set5KHCJLHSQSZHTZ = [
            PLCComand("6032.6032.LC5_YT603_2A_TASK1STOP", "boolean", "True", "", "", ""),  # 切丝段停止
            PLCComand("YG.YG.YG_YT630_OUTTASK2START", "boolean", "True", "", "", ""),  # 叶柜出柜任务2启动
            PLCComand("YG.YG.YG_YT630_OUTTASK2STOP", "boolean", "True", "", "", "")  # 叶柜出柜任务2结束
        ]

        # 3.5.	回潮/加料/烘丝:3.5.2.	设置组合停止：烘丝段组合停止
        self.Set5KHCJLHSHSZHTZ = [
            PLCComand("YSG.YSG.YSG_YT632_9000INGROUPSTARTING", "boolean", "True", "", "", ""),  # 9000线进柜组合启动
            PLCComand("YSG.YSG.YSG_YT632_3000INGROUPSTARTING", "boolean", "True", "", "", ""),  # 3000线进柜组合启动
            PLCComand("6032.6032.LD5_YT603_2B_TASK2STOP", "boolean", "True", "", "", ""),  # 烘丝段结束
            PLCComand("YSG.YSG.YSG_YT632_INTASK2STOP", "boolean", "True", "", "", "")  # 叶丝柜进柜任务2结束
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:回潮预热启动
        self.Set5KHCJLHSSZYRHCYRQD = [
            PLCComand("6022.6022.LAP5_WQ2116_PREHEAT_START", "boolean", "True", "", "", ""),  # 回潮预热启动
            PLCComand("6022.6022.LAP5_WQ2116_PREHEAT_STOP", "boolean", "False", "", "", "")  # 预热结束
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:回潮预热停止
        self.Set5KHCJLHSSZYRHCYRTZ = [
            PLCComand("6022.6022.LAP5_WQ2116_PREHEAT_START", "boolean", "False", "", "", ""),  # 回潮预热启动
            PLCComand("6022.6022.LAP5_WQ2116_PREHEAT_STOP", "boolean", "True", "", "", "")  # 预热结束
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:加料预热启动
        self.Set5KHCJLHSSZYRJLYRQD = [
            PLCComand("6022.6022.LAC5_SJ2147_PREHEAT_START", "boolean", "True", "", "", ""),  # 加料预热启动
            PLCComand("6022.6022.LAC5_SJ2147_PREHEAT_STOP", "boolean", "False", "", "", "")  # 预热结束
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:加料预热停止
        self.Set5KHCJLHSSZYRJLYRTZ = [
            PLCComand("6022.6022.LAC5_SJ2147_PREHEAT_START", "boolean", "False", "", "", ""),  # 加料预热停止
            PLCComand("6022.6022.LAC5_SJ2147_PREHEAT_STOP", "boolean", "True", "", "", "")  # 预热结束
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	5000烘丝机控制模式判断
        self.Read5KHCJLHSKZMS = [
            PLCComand("", "", "", "5H.5H.LD5_KL2226_ControlMode", "Byte", "4")  # 5000烘丝机控制模式
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:SIROX预热启动
        self.Set5KHCJLHSSZYRSIROXYRQD = [
            PLCComand("5H.5H.LD5_SX2224_SX1STARTPREHEATFK", "boolean", "False", "", "", "")  # SIROX预热启动
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:SIROX预热停止
        self.Set5KHCJLHSSZYRSIROXYRTZ = [
            PLCComand("5H.5H.LD5_SX2224_SX1STOPPREHEATFK", "boolean", "True", "", "", "")  # SIROX预热停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:烘丝预热启动
        self.Set5KHCJLHSSZYRHSYRQD = [
            PLCComand("5H.5H.LD5_KL2226_TT1STARTPREHEATFK", "boolean", "True", "", "", "")  # 烘丝预热启动
        ]

        # 3.5.	回潮/加料/烘丝:3.5.3.	设置预热:烘丝预热停止
        self.Set5KHCJLHSSZYRHSYRTZ = [
            PLCComand("5H.5H.LD5_KL2226_TT1STOPPREHEATFK", "boolean", "True", "", "", "")  # 烘丝预热停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.4.	设置冷却:烘丝冷却开始
        self.Set5KHCJLHSSZLQHSLQKS = [
            PLCComand("5H.5H.LD5_KL2226_TT1STARTCOOLDOWNFK", "boolean", "True", "", "", "")  # 烘丝冷却开始
        ]

        # 3.5.	回潮/加料/烘丝:3.5.4.	设置冷却:烘丝冷却停止
        self.Set5KHCJLHSSZLQHSLQTZ = [
            PLCComand("5H.5H.LD5_SX2224_SX1STARTCOOLDOWNFK", "boolean", "True", "", "", "")  # 烘丝冷却停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.4.	设置冷却:SIROX冷却开始
        self.Set5KHCJLHSSZLQSIROXLQKS = [
            PLCComand("5H.5H.LD5_SX2224_SX1STARTCOOLDOWNFK", "boolean", "True", "", "", "")  # SIROX冷却开始
        ]

        # 3.5.	回潮/加料/烘丝:3.5.4.	设置冷却:SIROX冷却停止
        self.Set5KHCJLHSSZLQSIROXLQTZ = [
            PLCComand("5H.5H.LD5_SX2224_SX1STOPCOOLDOWNFK", "boolean", "True", "", "", "")  # SIROX冷却停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.5.	设置生产停止:烘丝机生产开始
        self.Set5KHCJLHSSCTZHSSCKS = [
            PLCComand("5H.5H.LD5_KL2226_Line1Start", "boolean", "", "", "", "")  # 烘丝生产开始
        ]

        # 3.5.	回潮/加料/烘丝:3.5.5.	设置生产停止:烘丝机生产停止
        self.Set5KHCJLHSSCTZHSSCTZ = [
            PLCComand("5H.5H.LD5_KL2226_Line1Stop", "boolean", "True", "", "", "")  # 烘丝机生产停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.5.	设置生产停止:烘丝机生产暂停
        self.Set5KHCJLHSSCTZHSSCZT = [
            PLCComand("5H.5H.LD5_KL2226_GLPaus", "boolean", "True", "", "", "")  # 烘丝机生产暂停
        ]

        # 3.5.	回潮/加料/烘丝:3.5.6.	设置复位
        self.Set5KHCJLHSSZFW = [
            PLCComand("6022.6022.LAP5_YT602_2_CONTROL_YT6022_RESET", "boolean", "True", "", "", "")  # YT6022总故障复位
        ]

        # 3.5.	回潮/加料/烘丝:3.5.7.	设置清洗:SIROX清洁开始
        self.Set5KHCJLHSSZQXSIROXQJKS = [
            PLCComand("5H.5H.LD5_SX2224_SX1STARTCLEANINGFK", "boolean", "True", "", "", "")  # SIROX清洁开始
        ]

        # 3.5.	回潮/加料/烘丝:3.5.7.	设置清洗:SIROX清洁停止
        self.Set5KHCJLHSSZQXSIROXQJTZ = [
            PLCComand("5H.5H.LD5_SX2224_SX1STOPCLEANINGFK ", "boolean", "True", "", "", "")  # SIROX清洁停止
        ]

        # 2.4.	回潮模型加水系数:智能模型输出加水系数
        self.Set5KHSHCJSXS = [
            PLCComand("6022.6022.LAP5_WQ2116_AddWaterRatio1_SET", "float", "{0}", "", "", ""),  # 智能模型输出加水系数
            PLCComand("6022.6022.LAP5_3375_ROAD_S", "int", "{1}", "", "", ""),  # 智能模型输出（入口）
            PLCComand("6022.6022.LAP5_2118_ROAD_S", "int", "{2}", "", "", ""),  # 智能模型输出（出口）
            PLCComand("6022.6022.LAP5_WQ2116_OutMois_SP", "float", "{3}", "", "", "")  # 智能模型输出
        ]

        # 2.5.	加料模型加水系数:智能模型输出加料模型加水系数
        self.Set5KHSJLJSXS = [
            PLCComand("6022.6022.LAC5_SJ2147_AddWaterRatio1_SET", "float", "{0}", "", "", ""),  # 加料模型加水系数
            PLCComand("6022.6022.LAC5_SJ2147_OutMois_SP", "float", "{1}", "", "", "")  # 智能模型输出
        ]

        # 2.5.	加料模型加水比例:智能模型输出加料模型加水比例
        self.Set5KHSJLJSBL = [
            PLCComand("6022.6022.LAC5_SJ2147_AddWaterRatio2_SET", "float", "{0}", "", "", "")  # 料模型加水比例
        ]

        # 3.1.	烟叶回潮:3.1.8.	设置预配柜选柜打自动
        self.Set5KYYHCYPGXGDZD = [
            PLCComand("6022.6022.LAP5_YT602_2_R01_AUTO", "boolean", "True", "", "", ""),  # 开包机路径自动
            PLCComand("6022.6022.LAP5_YT602_2_R01_MANUAL", "boolean", "False", "", "", ""),  # 开包机手动
            PLCComand("6022.6022.LAP5_YT602_2_R01_LOCK", "boolean", "False", "", "", ""),  # 开包机闭锁
            PLCComand("6022.6022.LAP5_YT602_2_R02_AUTO", "boolean", "True", "", "", ""),  # 分切机路径自动
            PLCComand("6022.6022.LAP5_YT602_2_R02_MANUAL", "boolean", "False", "", "", ""),  # 分切机手动
            PLCComand("6022.6022.LAP5_YT602_2_R02_LOCK", "boolean", "False", "", "", ""),  # 分切机闭锁
            PLCComand("6022.6022.LAP5_YT602_2_R03_AUTO", "boolean", "True", "", "", ""),  # R03自动模式
            PLCComand("6022.6022.LAP5_YT602_2_R03_MANUAL", "boolean", "False", "", "", ""),  # R03手动
            PLCComand("6022.6022.LAP5_YT602_2_R03_LOCK", "boolean", "False", "", "", ""),  # R03闭锁
            PLCComand("6022.6022.LAP5_YT602_2_R04_AUTO", "boolean", "True", "", "", ""),  # R04自动模式
            PLCComand("6022.6022.LAP5_YT602_2_R04_MANUAL", "boolean", "False", "", "", ""),  # R04手动
            PLCComand("6022.6022.LAP5_YT602_2_R04_LOCK", "boolean", "False", "", "", ""),  # R04闭锁
            PLCComand("6022.6022.LAP5_YT602_2_R05_AUTO", "boolean", "True", "", "", ""),  # R05自动模式
            PLCComand("6022.6022.LAP5_YT602_2_R05_MANUAL", "boolean", "False", "", "", ""),  # R05手动
            PLCComand("6022.6022.LAP5_YT602_2_R05_LOCK", "boolean", "False", "", "", ""),  # R05闭锁
            PLCComand("6022.6022.LAP5_YT602_2_R06_AUTO", "boolean", "True", "", "", ""),  # R06自动模式
            PLCComand("6022.6022.LAP5_YT602_2_R06_MANUAL", "boolean", "False", "", "", ""),  # R06手动
            PLCComand("6022.6022.LAP5_YT602_2_R06_LOCK", "boolean", "False", "", "", ""),  # R06闭锁
            PLCComand("6022.6022.LAP5_YT602_2_R09_AUTO", "boolean", "True", "", "", ""),  # R09自动模式
            PLCComand("6022.6022.LAP5_YT602_2_R09_MANUAL", "boolean", "False", "", "", ""),  # R09手动
            PLCComand("6022.6022.LAP5_YT602_2_R09_LOCK", "boolean", "False", "", "", ""),  # R09闭锁
            PLCComand("6022.6022.LAP5_YT602_2_A01_AUTO", "boolean", "True", "", "", ""),  # YP06YP0自动模式
            PLCComand("6022.6022.LAP5_YT602_2_A01_MANUAL", "boolean", "False", "", "", ""),  # YP06YP0手动
            PLCComand("6022.6022.LAP5_YT602_2_A01_LOCK", "boolean", "False", "", "", ""),  # YP06YP0闭锁
            PLCComand("6022.6022.LAP5_YT602_2_A02_AUTO", "boolean", "True", "", "", ""),  # YP04YP0自动模式
            PLCComand("6022.6022.LAP5_YT602_2_A03_MANUAL", "boolean", "False", "", "", ""),  # YP04YP0手动
            PLCComand("6022.6022.LAP5_YT602_2_A02_LOCK", "boolean", "False", "", "", "")  # YP04YP0闭锁
        ]

        # 3.1.	烟叶回潮:3.1.8.	设置预配柜选柜打自动--路径选择
        self.Set5KYYHCYPGLJXZ = [
            PLCComand("6022.6022.LAP5_YT602_2_TASK1_R06", "boolean", "True", "", "", ""),  # 薄片旁线选中
            PLCComand("6022.6022.LAP5_YT602_2_TASK1_R09", "boolean", "True", "", "", "")  # 真空选中
        ]

        # 3.3.	加料:3.3.16.	设置预配柜选柜打自动
        self.Set5KJLYPGXGDZD = [
            PLCComand("6022.6022.LAP5_YT602_2_B01_AUTO", "boolean", "True", "", "", ""),  # YP07自动
            PLCComand("6022.6022.LAP5_YT602_2_B01_MANUAL", "boolean", "False", "", "", ""),  # YP07手动
            PLCComand("6022.6022.LAP5_YT602_2_B01_LOCK", "boolean", "False", "", "", ""),  # YP07闭锁
            PLCComand("6022.6022.LAP5_YT602_2_B02_AUTO", "boolean", "True", "", "", ""),  # YP06
            PLCComand("6022.6022.LAP5_YT602_2_B02_MANUAL", "boolean", "False", "", "", ""),  # YP06
            PLCComand("6022.6022.LAP5_YT602_2_B02_LOCK", "boolean", "False", "", "", ""),  # YP06
            PLCComand("6022.6022.LAP5_YT602_2_B03_AUTO", "boolean", "True", "", "", ""),  # YP05
            PLCComand("6022.6022.LAP5_YT602_2_B03_MANUAL", "boolean", "False", "", "", ""),  # YP05
            PLCComand("6022.6022.LAP5_YT602_2_B03_LOCK", "boolean", "False", "", "", ""),  # YP05
            PLCComand("6022.6022.LAP5_YT602_2_B04_AUTO", "boolean", "True", "", "", ""),  # YP04
            PLCComand("6022.6022.LAP5_YT602_2_B04_MANUAL", "boolean", "False", "", "", ""),  # YP04
            PLCComand("6022.6022.LAP5_YT602_2_B04_LOCK", "boolean", "False", "", "", ""),  # YP04
            PLCComand("6022.6022.LAP5_YT602_2_R07_AUTO", "boolean", "True", "", "", ""),  # R07
            PLCComand("6022.6022.LAP5_YT602_2_R07_MANUAL", "boolean", "False", "", "", ""),  # R07
            PLCComand("6022.6022.LAP5_YT602_2_R07_LOCK", "boolean", "False", "", "", ""),  # R07
            PLCComand("6022.6022.LAP5_YT602_2_R08_AUTO", "boolean", "True", "", "", ""),  # R08
            PLCComand("6022.6022.LAP5_YT602_2_R08_MANUAL", "boolean", "False", "", "", ""),  # R08
            PLCComand("6022.6022.LAP5_YT602_2_R08_LOCK", "boolean", "False", "", "", "")  # R08
        ]

        # 3.3.	加料:3.3.18.	设置叶柜入柜状态自动
        self.Set5KJLYGRGZTZD = [
            PLCComand("YG.YG.YG_YT630_INTASK2ALLAUTO", "boolean", "True", "", "", ""),  #
            PLCComand("YG.YG.YG_YT630_INTASK2ALLMANU", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_3_INA04AUTO", "boolean", "True", "", "", ""),  # R04
            PLCComand("YG.YG.YG_1129_3_INA04MANUAL", "boolean", "False", "", "", ""),  # R04
            PLCComand("YG.YG.YG_1129_3_INA04LOCK", "boolean", "False", "", "", ""),  # R04
            PLCComand("YG.YG.YG_1129_4_INA05AUTO", "boolean", "True", "", "", ""),  # R05
            PLCComand("YG.YG.YG_1129_4_INA05MANUAL", "boolean", "False", "", "", ""),  # R05
            PLCComand("YG.YG.YG_1129_4_INA05LOCK", "boolean", "False", "", "", ""),  # R05
            PLCComand("YG.YG.YG_1129_5_INA06AUTO", "boolean", "True", "", "", ""),  # R06
            PLCComand("YG.YG.YG_1129_5_INA06MANUAL", "boolean", "False", "", "", ""),  # R06
            PLCComand("YG.YG.YG_1129_5_INA06LOCK", "boolean", "False", "", "", ""),  # R06
            PLCComand("YG.YG.YG_1129_6_INA07AUTO", "boolean", "True", "", "", ""),  # R07
            PLCComand("YG.YG.YG_1129_6_INA07MANUAL", "boolean", "False", "", "", ""),  # R07
            PLCComand("YG.YG.YG_1129_6_INA07LOCK", "boolean", "False", "", "", ""),  # R07
            PLCComand("YG.YG.YG_1129_7_INA08AUTO", "boolean", "True", "", "", ""),  # R08
            PLCComand("YG.YG.YG_1129_7_INA08MANUAL", "boolean", "False", "", "", ""),  # R08
            PLCComand("YG.YG.YG_1129_7_INA08LOCK", "boolean", "False", "", "", ""),  # R08
            PLCComand("YG.YG.YG_1129_INA09AUTO", "boolean", "True", "", "", ""),  # R09
            PLCComand("YG.YG.YG_1129_INA09MANUAL", "boolean", "False", "", "", ""),  # R09
            PLCComand("YG.YG.YG_1129_INA09LOCK", "boolean", "False", "", "", ""),  # R09
            PLCComand("YG.YG.YG_1129_1_INA10AUTO", "boolean", "True", "", "", ""),  # R10
            PLCComand("YG.YG.YG_1129_1_INA10MANUAL", "boolean", "False", "", "", ""),  # R10
            PLCComand("YG.YG.YG_1129_1_INA10LOCK", "boolean", "False", "", "", ""),  # R10
            PLCComand("YG.YG.YG_1129_2_INA11AUTO", "boolean", "True", "", "", ""),  # R11
            PLCComand("YG.YG.YG_1129_2_INA11MANUAL", "boolean", "False", "", "", ""),  # R11
            PLCComand("YG.YG.YG_1129_2_INA11LOCK", "boolean", "False", "", "", ""),  # R11
            PLCComand("YG.YG.YG_1129_5_INA14AUTO", "boolean", "True", "", "", ""),  # R14
            PLCComand("YG.YG.YG_1129_5_INA14MANUAL", "boolean", "False", "", "", ""),  # R14
            PLCComand("YG.YG.YG_1129_5_INA14LOCK", "boolean", "False", "", "", ""),  # R14
            PLCComand("YG.YG.YG_1129_INA01AUTO", "boolean", "True", "", "", ""),  # YG2
            PLCComand("YG.YG.YG_1129_INA01MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_INA01LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_1_INA02AUTO", "boolean", "True", "", "", ""),  # YG4
            PLCComand("YG.YG.YG_1129_1_INA02MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_1_INA02LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_2_INA03AUTO", "boolean", "True", "", "", ""),  # YG6
            PLCComand("YG.YG.YG_1129_2_INA03MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_2_INA03LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_3_INA04AUTO", "boolean", "True", "", "", ""),  # YG8
            PLCComand("YG.YG.YG_1129_3_INA04MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_3_INA04LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_4_INA05AUTO", "boolean", "True", "", "", ""),  # YG10
            PLCComand("YG.YG.YG_1129_4_INA05MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_4_INA05LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_5_INA06AUTO", "boolean", "True", "", "", ""),  # YG12
            PLCComand("YG.YG.YG_1129_5_INA06MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_5_INA06LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_6_INA07AUTO", "boolean", "True", "", "", ""),  # YG14
            PLCComand("YG.YG.YG_1129_6_INA07MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_6_INA07LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_7_INA08AUTO", "boolean", "True", "", "", ""),  # YG16
            PLCComand("YG.YG.YG_1129_7_INA08MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_7_INA08LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_INA09AUTO", "boolean", "True", "", "", ""),  # YG1
            PLCComand("YG.YG.YG_1129_INA09MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_INA09LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_1_INA10AUTO", "boolean", "True", "", "", ""),  # YG3
            PLCComand("YG.YG.YG_1129_1_INA10MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_1_INA10LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_2_INA11AUTO", "boolean", "True", "", "", ""),  # YG5
            PLCComand("YG.YG.YG_1129_2_INA11MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_2_INA11LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_3_INA12AUTO", "boolean", "True", "", "", ""),  # YG7
            PLCComand("YG.YG.YG_1129_3_INA12MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_3_INA12LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_4_INA13AUTO", "boolean", "True", "", "", ""),  # YG9
            PLCComand("YG.YG.YG_1129_4_INA13MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_4_INA13LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_5_INA14AUTO", "boolean", "True", "", "", ""),  # YG11
            PLCComand("YG.YG.YG_1129_5_INA14MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_5_INA14LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_6_INA15AUTO", "boolean", "True", "", "", ""),  # YG13
            PLCComand("YG.YG.YG_1129_6_INA15MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_6_INA15LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_7_INA16AUTO", "boolean", "True", "", "", ""),  # YG15
            PLCComand("YG.YG.YG_1129_7_INA16MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1129_7_INA16LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1153A_INA17AUTO", "boolean", "True", "", "", ""),  # YG18
            PLCComand("YG.YG.YG_1153A_INA17MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1153A_INA17LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1153_1A_INA18AUTO", "boolean", "True", "", "", ""),  # YG20
            PLCComand("YG.YG.YG_1153_1A_INA18MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1153_1A_INA18LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1153_2A_INA19AUTO", "boolean", "True", "", "", ""),  # YG22
            PLCComand("YG.YG.YG_1153_2A_INA19MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1153_2A_INA19LOCK", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1153_3A_INA20AUTO", "boolean", "True", "", "", ""),  # YG24
            PLCComand("YG.YG.YG_1153_3A_INA20MANUAL", "boolean", "False", "", "", ""),  #
            PLCComand("YG.YG.YG_1153_3A_INA20LOCK", "boolean", "False", "", "", "")  #
        ]

        # 3.3.	加料:3.3.19.	设置叶柜选择路径
        self.Set5KJLYGXZLJ = [
            PLCComand("YG.YG.YG_1129_INTASK2A01", "boolean", "True", "", "", ""),  # YG02选中
            PLCComand("YG.YG.YG_1129_2_INTASK2A03", "boolean", "True", "", "", ""),  # YG04选中
            PLCComand("YG.YG.YG_1129_1_INTASK2A02", "boolean", "True", "", "", ""),  # YG06选中
            PLCComand("YG.YG.YG_1129_3_INTASK2A04", "boolean", "True", "", "", ""),  # YG08选中
            PLCComand("YG.YG.YG_1129_4_INTASK2A05", "boolean", "True", "", "", ""),  # YG10选中
            PLCComand("YG.YG.YG_1129_5_INTASK2A06", "boolean", "True", "", "", ""),  # YG12选中
            PLCComand("YG.YG.YG_1129_6_INTASK2A07", "boolean", "True", "", "", ""),  # YG14选中
            PLCComand("YG.YG.YG_1129_7_INTASK2A08", "boolean", "True", "", "", ""),  # YG16选中
            PLCComand("YG.YG.YG_1129_INTASK2A09", "boolean", "True", "", "", ""),  # YG01选中
            PLCComand("YG.YG.YG_1129_INTASK2A10", "boolean", "True", "", "", ""),  # YG03选中
            PLCComand("YG.YG.YG_1129_INTASK2A11", "boolean", "True", "", "", ""),  # YG05选中
            PLCComand("YG.YG.YG_1129_INTASK2A12", "boolean", "True", "", "", ""),  # YG07选中
            PLCComand("YG.YG.YG_1129_INTASK2A13", "boolean", "True", "", "", ""),  # YG09选中
            PLCComand("YG.YG.YG_1129_INTASK2A14", "boolean", "True", "", "", ""),  # YG11选中
            PLCComand("YG.YG.YG_1129_INTASK2A15", "boolean", "True", "", "", ""),  # YG13选中
            PLCComand("YG.YG.YG_1129_INTASK2A16", "boolean", "True", "", "", ""),  # YG15选中
            PLCComand("YG.YG.YG_1153A_INTASK2A17", "boolean", "True", "", "", ""),  # YG18选中
            PLCComand("YG.YG.YG_1153A_INTASK2A18", "boolean", "True", "", "", ""),  # YG20选中
            PLCComand("YG.YG.YG_1153A_INTASK2A19", "boolean", "True", "", "", ""),  # YG22选中
            PLCComand("YG.YG.YG_1153A_INTASK2A20", "boolean", "True", "", "", "")  # YG24选中
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-加料生产提升
        self.Set5KHCJLHSSZSCTSJLSCTS = [
            PLCComand("6022.6022.LAC5_WP2141_ASEND", "boolean", "1", "", "", "")  # 加料生产提升
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-加料强制停止
        self.Set5KHCJLHSSZSCTSLQZTZ = [
            PLCComand("6022.6022.LAC5_WP2141_FORCESTOP", "boolean", "1", "", "", "")  # 加料强制停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-加料强制提升
        self.Set5KHCJLHSSZSCTSLQZTS = [
            PLCComand("6022.6022.LAC5_WP2141_FORCEASEND", "boolean", "1", "", "", "")  # 加料强制提升
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-薄片回潮生产提升
        self.Set5KHCJLHSSZSCTSBPHCSCTS = [
            PLCComand("6022.6022.LAP5_2122_ASEND", "boolean", "1", "", "", "")  # 薄片回潮生产提升
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-薄片回潮强制停止
        self.Set5KHCJLHSSZSCTSBPHCQZTZ = [
            PLCComand("6022.6022.LAP5_2122_FORCESTOP", "boolean", "1", "", "", "")  # 薄片回潮强制停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-薄片回潮强制提升
        self.Set5KHCJLHSSZSCTSBPHCQZTS = [
            PLCComand("6022.6022.LAP5_2122_FORCEASEND", "int", "1", "", "", "")  # 薄片回潮强制提升
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-回潮生产提升
        self.Set5KHCJLHSSZSCTSHCSCTS = [
            PLCComand("6022.6022.LVR_3373_ASEND", "boolean", "1", "", "", "")  # 回潮生产提升
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-回潮强制停止
        self.Set5KHCJLHSSZSCTSHCQZTZ = [
            PLCComand("6022.6022.LVR_3373_FORCESTOP", "boolean", "1", "", "", "")  # 回潮强制停止
        ]

        # 3.5.	回潮/加料/烘丝:3.5.8.	设置生产提升-回潮强制提升
        self.Set5KHCJLHSSZSCTSHCQZTS = [
            PLCComand("6022.6022.LVR_3373_FORCEASEND", "boolean", "1", "", "", "")  # 回潮强制提升
        ]

    def Get_Command(self, commandtype):
        commandbody = ""

        if commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL:
            commandbody = PLCComandList(self.Set5KTempAll).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_RESET_ALL:
            commandbody = PLCComandList(self.ReSet5KTempAll).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_READ_HMI:
            commandbody = PLCComandList(self.Read5KTBWDTempHMI).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_READ_TEMPS:
            commandbody = PLCComandList(self.Read5KTBWDTemps).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SET_TEMPS:
            commandbody = PLCComandList(self.Set5KTBWDTemps).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SET_TIC_COS:
            commandbody = PLCComandList(self.Set5KTBWDTTICCOS).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_MAN:
            commandbody = PLCComandList(self.Set5KTBWDSwitchMan).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_AUTO:
            commandbody = PLCComandList(self.Set5KTBWDSwitchAuto).toJSON()

        elif commandtype == DeviceCommandTypes.SIM_TEST_D1_T1:
            commandbody = PLCComandList(self.SetSIMTestD1T1).toJSON()

        elif commandtype == DeviceCommandTypes.SIM_TEST_D1_T2:
            commandbody = PLCComandList(self.SetSIMTestD1T2).toJSON()

        elif commandtype == DeviceCommandTypes.SIM_TEST_D1_T4:
            commandbody = PLCComandList(self.SetSIMTestD1T4).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL:
            commandbody = PLCComandList(self.SetLD5Test).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL:
            commandbody = PLCComandList(self.ReSetLD5Test).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_YYHCLL:
            commandbody = PLCComandList(self.Set5KYYHCLL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_SFYTDH:
            commandbody = PLCComandList(self.Set5KYYHCSFYTDH).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_RFWD:
            commandbody = PLCComandList(self.Set5KYYHCRFWD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_RKSF:
            commandbody = PLCComandList(self.Set5KYYHCRKSF).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_CKSF:
            commandbody = PLCComandList(self.Set5KYYHCCKSF).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_JSXS:
            commandbody = PLCComandList(self.Set5KYYHCJSXS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_ZKDZCLL:
            commandbody = PLCComandList(self.Set5KYYHCZKDZCLL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_BPHCLL:
            commandbody = PLCComandList(self.Set5KBPHCBPHCLL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_SFYTDH:
            commandbody = PLCComandList(self.Set5KBPHCSFYTDH).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_RFWD:
            commandbody = PLCComandList(self.Set5KBPHCRFWD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_RKSF:
            commandbody = PLCComandList(self.Set5KBPHCRKSF).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_CKSF:
            commandbody = PLCComandList(self.Set5KBPHCCKSF).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_JSBL:
            commandbody = PLCComandList(self.Set5KBPHCJSBL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_CKSFSX:
            commandbody = PLCComandList(self.Set5KBPHCCKSFSX).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_CKSFXX:
            commandbody = PLCComandList(self.Set5KBPHCCKSFXX).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_CKSFBL:
            commandbody = PLCComandList(self.Set5KBPHCCKSFBL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_TLJSL:
            commandbody = PLCComandList(self.Set5KBPHCTLJSL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_WLJSL:
            commandbody = PLCComandList(self.Set5KBPHCWLJSL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_YYLL:
            commandbody = PLCComandList(self.Set5KJLYYLL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_SFYTDH:
            commandbody = PLCComandList(self.Set5KJLSFYTDH).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_JLBL:
            commandbody = PLCComandList(self.Set5KJLJLBL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_RFWD:
            commandbody = PLCComandList(self.Set5KJLRFWD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LT1WD:
            commandbody = PLCComandList(self.Set5KJLLT1WD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LT2WD:
            commandbody = PLCComandList(self.Set5KJLLT2WD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_TLXZXS:
            commandbody = PLCComandList(self.Set5KJLTLXZXS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_WLXZXS:
            commandbody = PLCComandList(self.Set5KJLWLXZXS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_WLLL:
            commandbody = PLCComandList(self.Set5KJLWLLL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_JSXS:
            commandbody = PLCComandList(self.Set5KJLJSXS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_JSBL:
            commandbody = PLCComandList(self.Set5KJLJSBL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_CKHSL:
            commandbody = PLCComandList(self.Set5KJLCKHSL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_DSFW:
            commandbody = PLCComandList(self.Set5KJLDSFW).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_READ_YLAN:
            commandbody = PLCComandList(self.Read5KJLYLAN).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LTXZ1:
            commandbody = PLCComandList(self.Set5KJLLTXZ1).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LTXZ2:
            commandbody = PLCComandList(self.Set5KJLLTXZ2).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_READ_LYAN:
            commandbody = PLCComandList(self.Read5KJLLYAN).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LYDM1:
            commandbody = PLCComandList(self.Set5KJLLYDM1).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LYDM2:
            commandbody = PLCComandList(self.Set5KJLLYDM2).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_LLLL:
            commandbody = PLCComandList(self.Set5KHSHSLLLL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_SIROX:
            commandbody = PLCComandList(self.Set5KHSHSSIROX).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_RFFS:
            commandbody = PLCComandList(self.Set5KHSHSRFFS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_RFWD:
            commandbody = PLCComandList(self.Set5KHSHSRFWD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_CKSF:
            commandbody = PLCComandList(self.Set5KHSHSCKSF).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_SFYTD:
            commandbody = PLCComandList(self.Set5KHSHSSFYTD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_TSSD:
            commandbody = PLCComandList(self.Set5KHSHSSTSSD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_HCZHQD:
            commandbody = PLCComandList(self.Set5KHCJLHSHCZHQD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT1XZ:
            commandbody = PLCComandList(self.Read5KHCJLHSJLZHQDLT1XZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT1ZL:
            commandbody = PLCComandList(self.Read5KHCJLHSJLZHQDLT1ZL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT2XZ:
            commandbody = PLCComandList(self.Read5KHCJLHSJLZHQDLT2XZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT2ZL:
            commandbody = PLCComandList(self.Read5KHCJLHSJLZHQDLT2ZL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_JLZHQD:
            commandbody = PLCComandList(self.Set5KHCJLHSJLZHQD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_QSZHQD:
            commandbody = PLCComandList(self.Set5KHCJLHSQSZHQD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_HSZHQD:
            commandbody = PLCComandList(self.Set5KHCJLHSHSZHQD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_HCZHTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSHCZHTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_JLZHTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSJLZHTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_QSZHTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSQSZHTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_HSZHTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSHSZHTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_HCYRQD:
            commandbody = PLCComandList(self.Set5KHCJLHSSZYRHCYRQD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_HCYRTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZYRHCYRTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_JLYRQD:
            commandbody = PLCComandList(self.Set5KHCJLHSSZYRJLYRQD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_JLYRTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZYRJLYRTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_KZMS:
            commandbody = PLCComandList(self.Read5KHCJLHSKZMS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_SIROX_YRQD:
            commandbody = PLCComandList(self.Set5KHCJLHSSZYRSIROXYRQD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_SIROX_YRTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZYRSIROXYRTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_HSYRQD:
            commandbody = PLCComandList(self.Set5KHCJLHSSZYRHSYRQD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_HSYRTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZYRHSYRTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZLQ_HSLQKS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZLQHSLQKS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZLQ_HSLQTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZLQHSLQTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZLQ_SIROXLQKS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZLQSIROXLQKS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZLQ_SIROXLQTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZLQSIROXLQTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCKS:
            commandbody = PLCComandList(self.Set5KHCJLHSSCTZHSSCKS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSCTZHSSCTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCZT:
            commandbody = PLCComandList(self.Set5KHCJLHSSCTZHSSCZT).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZFW:
            commandbody = PLCComandList(self.Set5KHCJLHSSZFW).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZQX_SIROXQJKS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZQXSIROXQJKS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZQX_SIROXQJTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZQXSIROXQJTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_HC_SET_JSXS:
            commandbody = PLCComandList(self.Set5KHSHCJSXS).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_JL_SET_JSXS:
            commandbody = PLCComandList(self.Set5KHSJLJSXS).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_JL_SET_JSBL:
            commandbody = PLCComandList(self.Set5KHSJLJSBL).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_YPGXGDZD:
            commandbody = PLCComandList(self.Set5KYYHCYPGXGDZD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_YPGLJXZ:
            commandbody = PLCComandList(self.Set5KYYHCYPGLJXZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_YPGXGDZD:
            commandbody = PLCComandList(self.Set5KJLYPGXGDZD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_YGRGZTZD:
            commandbody = PLCComandList(self.Set5KJLYGRGZTZD).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_YGXZLJ:
            commandbody = PLCComandList(self.Set5KJLYGXZLJ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_JLSCTS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSJLSCTS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_JLQZTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSLQZTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_JLQZTS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSLQZTS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCSCTS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSBPHCSCTS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCQZTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSBPHCQZTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCQZTS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSBPHCQZTS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_HCSCTS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSHCSCTS).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_HCQZTZ:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSHCQZTZ).toJSON()

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_HCQZTS:
            commandbody = PLCComandList(self.Set5KHCJLHSSZSCTSHCQZTS).toJSON()

        else:
            commandbody = ""
            logger.error("Error: Wrong Command Type: " + commandtype)

        return commandbody
