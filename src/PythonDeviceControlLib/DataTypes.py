# -*- coding:UTF-8 -*-
import sys
import json


class PLCComand:
    def __init__(self, address, datatype, value):
        self.Address = address
        self.DataType = datatype
        self.Value = value

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class PLCComandList:
    def __init__(self, inputlist):
        self.commandList = inputlist

    def AddCommand(self, command):
        self.commandList.append(command)

    def toJSON(self):
        return json.dumps(self.commandList, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class PLCResults:
    def __init__(self, address, issetsucessful, value):
        self.Address = address
        self.IsSetSucessful = issetsucessful
        self.Message = value

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
