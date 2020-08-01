import json

from src.PythonDeviceControlLib.CommandHandler import CommandHandler
from src.PythonDeviceControlLib.DeviceCommands import DeviceCommandTypes
from src.config.config import CONTROL_URL

handler = CommandHandler(CONTROL_URL)


def _is_failure(res):
    fail = False
    for r in res:
        fail = fail or not r['IsSetSuccessful']
    return fail


def _get_value(res, address):
    for r in res:
        if address == r['Address']:
            return r['Value']
    return None


def set_prod(values: list):
    for _ in range(5):
        res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_READ_HMI, [])
        res = json.loads(res.decode())
        if not _is_failure(res) and int(_get_value(res, "5H.5H.LD5_KL2226_PID042MCVHMI")) == 1:
            res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL, values)
            res = json.loads(res.decode())
            return res
        handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_MAN, [])
    return None


def reset_prod(values: list):
    for _ in range(5):
        res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_READ_HMI, [])
        res = json.loads(res.decode())
        if not _is_failure(res) and int(_get_value(res, "5H.5H.LD5_KL2226_PID042MCVHMI")) == 0:
            res = handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_RESET_ALL, values)
            res = json.loads(res.decode())
            return res
        handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_AUTO, [])
    return None


def set_test(values: list):
    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL, values)
    res = json.loads(res.decode())
    return res


def reset_test(values: list):
    res = handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL, values)
    res = json.loads(res.decode())
    return res
