import json

import requests

from src.config.config import CONFIG_URL


def read_standard(brand, default_1, default_2):
    try:
        # 请求预热的配置
        body = {
            "BrandCode": brand,
            "WorkstageCode": "LD5",
            "TagReads": [
                "5H.5H.LD5_KL2226_TT1StandardTemp1",
                "5H.5H.LD5_KL2226_TT1StandardTemp2"
            ]
        }
        res = requests.post(CONFIG_URL, json=body)
        if (res.status_code / 100) == 2:
            json_obj = json.loads(res.text)
            rows = json_obj.get('data').get('Rows')
            standard_1 = default_1
            standard_2 = default_2
            for row in rows:
                if row.get('TagRead') == "5H.5H.LD5_KL2226_TT1StandardTemp1":
                    standard_1 = float(row.get('ParmSet')) - 3
                if row.get('TagRead') == "5H.5H.LD5_KL2226_TT1StandardTemp2":
                    standard_2 = float(row.get('ParmSet')) - 3
            return None, {
                'standard_1': standard_1,
                'standard_2': standard_2
            }
        else:
            return res.text, None
    except Exception as e:
        return str(e), None


rslt = read_standard('TG####A', 130, 120)
print(rslt)
