FROM python:3.6.9

ADD . /qingdao

WORKDIR /qingdao

RUN pip install --default-timeout=100 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN sed -i "s?^ROOT_PATH.*?ROOT_PATH = '/qingdao'?g" /qingdao/src/config/config.py

CMD export PYTHONPATH=:/qingdao && python3 /qingdao/src/model_controller.py  2>&1 | tee model_service.log