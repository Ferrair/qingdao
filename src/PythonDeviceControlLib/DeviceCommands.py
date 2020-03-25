# -*- coding:UTF-8 -*-
from enum import Enum

from src.PythonDeviceControlLib.DataTypes import PLCComand, PLCComandList


class DeviceCommandTypes(Enum):
    ML_5K_HS_TB_WD_SET_ALL = 0  # 设置五千线烘丝机筒壁1区、2区温度设定值
    ML_5K_HS_TB_WD_RESET_ALL = 1  # 恢复五千线烘丝机筒壁1区、2区温度设定值
    SIM_TEST_D1_T1 = 2  # 测试点位1,#ML
    SIM_TEST_D1_T2 = 3  # 测试点位2,#ML
    SIM_TEST_D1_T4 = 4  # 测试点位3,#APP
    ML_5H_5H_LD5_TEST_SET_ALL = 5  # 微软测试写入滚筒区域1\2温度设定
    ML_5H_5H_LD5_TEST_RESET_ALL = 6  # 微软测试写入滚筒区域1\2温度恢复

    # //ML_5K_HS_TB_WD_1, //2.1.	设置五千线烘丝机筒壁1区温度设定值
    # //ML_5K_HS_TB_WD_2,//2.2.	设置五千线烘丝机筒壁2区温度设定值
    # //3.1.1.	设置烟叶回潮流量
    # //3.1.2.	设置水分仪通道号
    # //3.1.3.	设置热风温度
    # //3.1.4.	设置入口出口水分
    # //3.1.5.	设置出口水分
    # //3.1.6.	设置加水系数
    # //3.1.7.	设置出口水分
    # //3.1.8.	设置真空电子秤流量

    # //3.2.1.	设置薄片回潮流量
    # //3.2.2.	设置水分仪通道号
    # //3.2.3.	设置热风温度
    # //3.2.4.	设置入口出口水分
    # //3.2.5.	设置出口水分
    # //3.2.6.	设置加水系数
    # //3.2.7.	设置出口水分
    # //3.2.8.	设置出口水分上限设定值
    # //3.2.9.	设置出口水分下限设定值
    # //3.2.10.	设置入口水分比例设定值
    # //3.2.11.	头料加水量
    # //3.2.12.	设置尾料加水量

    # //3.3.1.	设置烟叶流量
    # //3.3.2.	设置水分仪通道号
    # //3.3.3.	设置加料比例
    # //3.3.4.	设置热风温度
    # //3.3.5.	设置料桶1温度
    # //3.3.6.	设置料桶2温度
    # //3.3.7.	设置头料修正系数
    # //3.3.8.	设置尾料修正系数
    # //3.3.9.	设置尾料流量
    # //3.3.10.	设置加水系数
    # //3.3.11.	设置加水比例
    # //3.3.12.	设置出口含水量
    # //3.3.13.	设置带速范围
    # //3.3.14.	设置要料重量
    # //3.3.15.	设置料液代码

    # //3.4.1.	设置来料流量
    # //3.4.2.	设置SIROX蒸汽流量
    # //3.4.3.	设置1区筒壁温度（重复项）
    # //3.4.4.	设置2区筒壁温度（重复项）
    # //3.4.5.	设置热风风速
    # //3.4.6.	设置热风温度
    # //3.4.7.	设置出口水分
    # //3.4.8.	设置水分仪通道
    # //3.4.9.	设置脱水速度

    # //3.5.1.	设置组合启动
    # //3.5.2.	设置组合停止
    # //3.5.3.	设置预热
    # //3.5.4.	设置冷却
    # //3.5.5.	设置生产停止
    # //3.5.6.	设置复位
    # //3.5.7.	设置清洗
    # //3.5.8.	设置状态


class DeviceCommandGenerator:

    def __init__(self):
        self.Set5KTempAll = [
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_1", "float", "0"),
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_2", "float", "0"),
            PLCComand("5H.5H.LD5_KL2226_PID04_OPERATE1_MAN", "Boolean", "True"),
            PLCComand("5H.5H.LD5_KL2226_PID04_CV", "float", "0"),
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp1", "float", "{0}"),
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp2", "float", "{1}")
        ]

        self.ReSet5KTempAll = [
            PLCComand("5H.5H.LD5_KL2226_PID04_CV", "float", "1"),
            PLCComand("5H.5H.LD5_KL2226_PID04_OPERATE1_AUTO", "Boolean", "True"),
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp1", "float", "{0}"),
            PLCComand("5H.5H.LD5_KL2226_TT1StandardTemp2", "float", "{1}"),
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_2", "float", "1"),
            PLCComand("5H.5H.LD5_KL2226_TIC_CO_PP_1", "float", "1")
        ]

        self.SetSIMTestD1T1 = [
            PLCComand("test.d1.t1", "float", "{0}")
        ]

        self.SetSIMTestD1T2 = [
            PLCComand("test.d1.t2", "float", "{0}")
        ]

        self.SetSIMTestD1T4 = [
            PLCComand("test.d1.t4", "float", "{0}")
        ]

        self.SetLD5Test = [
            PLCComand("5H.5H.LD5_test1", "float", "{0}"),
            PLCComand("5H.5H.LD5_test2", "float", "{1}")
        ]

        self.ReSetLD5Test = [
            PLCComand("5H.5H.LD5_test1", "float", "{0}"),
            PLCComand("5H.5H.LD5_test2", "float", "{1}")
        ]

    def Get_Command(self, commandtype):
        commandbody = ""

        if commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL:
            commandbody = PLCComandList(self.Set5KTempAll).toJSON()

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_RESET_ALL:
            commandbody = PLCComandList(self.ReSet5KTempAll).toJSON()

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

        else:
            commandbody = ""
            logger.error("Error: Wrong Command Type:" + commandtype)

        return commandbody
