# -*- coding:UTF-8 -*-
import sys
import urllib.request
import json
from src.PythonDeviceControlLib.PLog import logger


def RestApiPost(url, values):
    try:
        headers = {'Content-type': 'application/json; charset=utf-8', 'Accept': 'application/json; charset=utf-8'}
        data = json.dumps(values)
        data = bytes(data, 'utf8')
        logger.info(data)
        req = urllib.request.Request(url, data, headers)
        response = urllib.request.urlopen(req)
        body = response.read()
        return body
    except:
        logger.error(u'REST API Failed: %s' % str(sys.exc_info()))

    return 'REST API Failed: %s' % str(sys.exc_info())


if __name__ == '__main__':
    url = "http://localhost:64035/api/PLCAPI"
    value = "[{\"Address\":\"test.d1.t1\",\"DataType\":\"float\",\"Value\":\"12\"}]"
    res = RestApiPost(url, value)
    print(res)
