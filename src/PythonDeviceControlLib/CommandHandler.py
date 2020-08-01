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

    def RunPLCCommand2(self, command):
        rtval = self.ExecuteRestApi(command)
        return rtval

    def RunPLCCommand(self, commandtype, args):
        rtval = ""
        generator = DeviceCommandGenerator()
        length = len(args)

        if commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_RESET_ALL and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_READ_HMI:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_READ_TEMPS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SET_TEMPS and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SET_TIC_COS and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_MAN:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_AUTO:
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

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_YYHCLL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_SFYTDH and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_RFWD and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_RKSF and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_CKSF and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_JSXS and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_ZKDZCLL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_BPHCLL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_SFYTDH and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_RFWD and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_RKSF and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_CKSF and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_JSBL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_CKSFSX and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_CKSFXX and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_CKSFBL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_TLJSL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_BPHC_SET_WLJSL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_YYLL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_SFYTDH and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_JLBL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_RFWD and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LT1WD and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LT2WD and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_TLXZXS and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_WLXZXS and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_WLLL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_JSXS and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_JSBL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_CKHSL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_DSFW and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_READ_YLAN:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LTXZ1 and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LTXZ2 and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_READ_LYAN:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LYDM1 and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_LYDM2 and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_LLLL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_SIROX and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_RFFS and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_RFWD and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_CKSF and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_SFYTD and length == 2:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HS_SET_TSSD and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_HCZHQD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT1XZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT1ZL:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT2XZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_JLZHQD_LT2ZL:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_JLZHQD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_QSZHQD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_HSZHQD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_HCZHTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_JLZHTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_QSZHTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_HSZHTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_HCYRQD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_HCYRTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_JLYRQD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_JLYRTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_READ_KZMS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_SIROX_YRQD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_SIROX_YRTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_HSYRQD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZYR_HSYRTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZLQ_HSLQKS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZLQ_HSLQTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZLQ_SIROXLQKS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZLQ_SIROXLQTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCKS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTZ_HSSCZT:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZFW:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZQX_SIROXQJKS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SZQX_SIROXQJTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_HC_SET_JSXS and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_JL_SET_JSXS and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ML_5K_HS_JL_SET_JSBL and length == 1:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_YPGXGDZD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_YYHC_SET_YPGLJXZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_YPGXGDZD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_YGRGZTZD:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_JL_SET_YGXZLJ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_JLSCTS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_JLQZTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_JLQZTS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCSCTS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCQZTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_BPHCQZTS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_HCSCTS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_HCQZTZ:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        elif commandtype == DeviceCommandTypes.ZZ_5K_HS_HCJLHS_SET_SCTS_HCQZTS:
            rtval = self.FormantAndRunCommand(generator.Get_Command(commandtype), args)

        else:
            logger.error("Error: Wrong Command, type:" + str(commandtype) + ",args: " + str(args))

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

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_READ_HMI, [])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_READ_TEMPS, [])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_TEMPS, ["110", "120"])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_TIC_COS, ["0", "0"])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_MAN, [])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_AUTO, [])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T1, ["12"])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T2, ["13"])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T4, ["14"])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL, ["120", "111"])
    print(res)

    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL, ["120", "111"])
    print(res)
