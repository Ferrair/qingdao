# -*- coding:UTF-8 -*-
import sys

from src.PythonDeviceControlLib.DeviceCommands import DeviceCommandGenerator, DeviceCommandTypes
from src.PythonDeviceControlLib.PLog import logger
from src.PythonDeviceControlLib.RestHandler import RestApiPost


class CommandHandler:
    def __init__(self, url):
        self.webApiUrl = url

    def ExecuteRestApi(self, command):
        res = RestApiPost(self.webApiUrl, command)
        return res

    def FormantAndRunCommand(self, command, args):
        rtval = command
        alength = len(args)
        if alength > 0:
            for index in range(0, alength):
                tempstr = "{" + str(index) + "}"
                rtval = rtval.replace(tempstr, args[index])
        logger.info(rtval)
        rtval = self.ExecuteRestApi(rtval)
        return rtval

    def RunPLCCommand(self, commandtype, args):
        rtval = ""
        generator = DeviceCommandGenerator()
        length = len(args)

        if commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_RESET_ALL and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.SIM_TEST_D1_T1 and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.SIM_TEST_D1_T2 and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.SIM_TEST_D1_T4 and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        else:
            logger.error("Error: Wrong Command,type:" + str(commandtype) + ",args:" + str(args))

        return rtval


if __name__ == '__main__':
    url = "http://localhost:64035/api/PLCAPI"
    handler = CommandHandler(url)
    # res = handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T1,["11"])
    # print(res)

    # res = handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T2,["12"])
    # print(res)

    # res = handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T4,["14"])
    # print(res)

    # res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL,["100","110"])
    # print(res)

    # res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_RESET_ALL,["105","105"])
    # print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL, ["105", "125"])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL, ["110", "120"])
    print(res)
