Python Device Control Lib: Lib interface between App & IOT HUB
Testing Environment:Windows 10/python 3.7.7

Sample(url is the IOT HUB REST API URL):

url = "http://localhost:64035/api/PLCAPI"
handler = CommandHandler(url)
res = handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T2,["11"])
print(res)


设备控制接口说明：

设备接口函数主要包括两个：

一、定制化命令接口函数：RunPLCCommand(self,commandtype,args)
   参数说明：
   commandtype：设备命令类型编号，每个编号对应一个具体的命令
   args：字符串数组类型的动态参数，参数示例：["110","120"]，传入两个参数值110和120

二、自定义命令接口函数：RunPLCCommand2(self,command)
    1.参数说明：
	string command： json格式的命令参数
	2.命令格式说明：
	Address:PLC点位标签地址（用于写入）
	DataType: PLC点位数据类型
	Value: PLC点位写入值（没有检查值时默认写入以后再读取比对）
	CheckAddress: PLC点位标签地址（用于读取检查）
	CheckDataType:PLC点位数据类型
	CheckValue:PLC点位检查值
	3.命令使用说明：
	情形1：写入一个点位，并且在写入以后读取该点位的值用来确认是否写入成功，这个时候只需要设置Address,DataType,Value三个值。
	情形2：写入一个点位，从另一个点位读取值判断是否写入成功：这个时候Address,DataType,Value用于写入，CheckAddress,CheckDataType,CheckValue用于读取对比，其中从CheckAddress读取的值必需跟CheckValue值一致才算成功。
	情形3：不写入点位，只读取点位的值：这个时候仅使用CheckAddress,CheckDataType,CheckValue，其中CheckValue可不填写。
	4.参数示例：
	[{"Address":"5H.5H.LD5_test1","DataType":"float","Value":"11","CheckAddress":null,"CheckDataType":null,"CheckValue":null},{"Address":"5H.5H.LD5_test2","DataType":"float","Value":"100","CheckAddress":null,"CheckDataType":null,"CheckValue":null}]
          

定制化命令详细说明：

 1.接口功能：设置五千线烘丝机筒壁1区、2区温度设定值（统一切换成Hauni手动模式和设置温度）
   命令类型编号：ML_5K_HS_TB_WD_SET_ALL
   参数说明：args[0]:筒壁1区温度设置值，  args[1]:筒壁2区温度设置值
   调用示例：res = await handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_ALL, new string[] { "11", "100" });

 2.接口功能：恢复五千线烘丝机筒壁1区、2区温度设定值（统一切换成Hauni自动模式和设置温度）
   命令类型编号：ML_5K_HS_TB_WD_RESET_ALL
   参数说明：args[0]:筒壁1区温度设置值，  args[1]:筒壁2区温度设置值
   调用示例：res = await handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_RESET_ALL, new string[] { "11", "100" });


 3.接口功能：读取HMI判断当前Huani工作状态是自动还是手动
   命令类型编号：ML_5K_HS_TB_WD_READ_HMI
   参数说明：无参数
   调用示例：res = await handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_READ_HMI,null);

 4.接口功能：读取当前五千线烘丝机筒壁1区、2区温度
   命令类型编号：ML_5K_HS_TB_WD_READ_TEMPS
   参数说明：无参数
   调用示例：res = await handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_READ_TEMPS, null);

 5.接口功能：设置当前五千线烘丝机筒壁1区、2区温度,不执行切换
   命令类型编号：ML_5K_HS_TB_WD_SET_TEMPS
   参数说明：args[0]:筒壁1区温度设置值，  args[1]:筒壁2区温度设置值
   调用示例：res = await handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_TEMPS, new string[] { "11", "100" });

 6.接口功能：设置当前五千线烘丝机筒壁1区、2区正向控制除水     
   命令类型编号：ML_5K_HS_TB_WD_SET_TIC_COS
   参数说明：args[0]:筒壁1区正向控制除水,0:Huani手动模式；1:Huani自动模式，  args[1]:筒壁2区正向控制除水,0:Huani手动模式；1:Huani自动模式
   调用示例：res = await handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SET_TIC_COS, new string[] { "0", "0" });

 7.接口功能：设置五千线烘丝机筒壁1区、2区温度,切换到Hauni手动模式
   命令类型编号：ML_5K_HS_TB_WD_SWITCH_MAN
   参数说明：无参数
   调用示例：res = await handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_MAN, null);

 8.接口功能：设置五千线烘丝机筒壁1区、2区温度,切换到Hauni自动模式    
   命令类型编号：ML_5K_HS_TB_WD_SWITCH_AUTO
   参数说明：无参数
   调用示例：res = await handler.RunPLCCommand(DeviceCommandTypes.ML_5K_HS_TB_WD_SWITCH_AUTO, null);


注：命令3,4,5,6,7,8为命令1、2的扩展，方便应用层分步控制


 9.接口功能：测试点位1设置值
   命令类型编号：SIM_TEST_D1_T1
   参数说明：args[0]:点位设置值
   调用示例：res =  await handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T1, new string[] { "12" });


 10.接口功能：测试点位2设置值
   命令类型编号：SIM_TEST_D1_T2
   参数说明：args[0]:点位设置值
   调用示例：res =  await handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T2, new string[] { "13" });


 11.接口功能：测试点位3设置值
   命令类型编号：SIM_TEST_D1_T4
   参数说明：args[0]:点位设置值
   调用示例：res =  await handler.RunPLCCommand(DeviceCommandTypes.SIM_TEST_D1_T4, new string[] { "14" });


 12.接口功能：PLC微软测试点位，写入滚筒区域1\2温度设定
   命令类型编号：ML_5H_5H_LD5_TEST_SET_ALL
   参数说明：args[0]:点位设置值
   调用示例：res =  await handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_SET_ALL, new string[] { "11", "100" });


 13.接口功能：PLC微软测试点位，写入滚筒区域1\2温度恢复
   命令类型编号：ML_5H_5H_LD5_TEST_RESET_ALL
   参数说明：args[0]:点位设置值
   调用示例：res =  await handler.RunPLCCommand(DeviceCommandTypes.ML_5H_5H_LD5_TEST_RESET_ALL, new string[] { "11", "100" });
