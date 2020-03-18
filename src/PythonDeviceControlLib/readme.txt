Python Device Control Lib: Lib interface between App & IOT HUB
Testing Environment:Windows 10/python 3.7.7

Sample(url is the IOT HUB REST API URL):

url = "http://localhost:64035/api/PLCAPI"
handler = CommandHandler(url)
res = handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T2,["11"])
print(res)
