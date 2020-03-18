# -*- coding:UTF-8 -*-
import sys
import urllib.request
from src.PythonDeviceControlLib.PLog import logger


def RestApiPost(url, values):
    try:
        headers = {'Content-type': 'application/json; charset=utf-8', 'Accept': 'application/json; charset=utf-8'}
        # logger.info(u'URL %s'%url)
        # logger.info(u'Headers: %s'%headers)
        # logger.info(u'Values: %s'%values)    
        req = urllib.request.Request(url, values, headers)
        response = urllib.request.urlopen(req)
        body = response.read()
        logger.info(body)
        return body
    except:
        logger.error(u'REST API Failed:%s' % str(sys.exc_info()))

    return "Error:REST API Failed."


if __name__ == '__main__':
    url = "http://localhost:64035/api/PLCAPI"
    value = "[{\"Address\":\"test.d1.t1\",\"DataType\":\"float\",\"Value\":\"12\"}]"
    res = RestApiPost(url, value)
    print(res)
