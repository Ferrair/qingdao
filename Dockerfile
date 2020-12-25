FROM python:3.6.9

ADD model_save /qingdao/model_save
ADD src /qingdao/src
ADD build.sh /qingdao/build.sh
ADD pid.txt /qingdao/pid.txt
ADD Dockerfile /qingdao/Dockerfile
ADD makefile /qingdao/makefile
ADD logs /qingdao/logs
ADD requirements.txt /qingdao/requirements.txt

WORKDIR /qingdao

RUN pip install --default-timeout=100 -r requirements.txt
RUN sed -i "s?^ROOT_PATH.*?ROOT_PATH = '/qingdao'?g" /qingdao/src/config/config.py

CMD export PYTHONPATH=:/qingdao && python3 /qingdao/src/model_controller.py  2>&1 | tee model_service.log