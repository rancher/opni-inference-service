FROM rancher/opni-python-base:3.8-torch

RUN zypper -n ref  && \
    zypper --non-interactive in wget && \
    zypper --non-interactive in zip

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./opni_inference_service/ /app/opni_inference_service/

RUN chmod a+rwx -R /app
WORKDIR /app/opni_inference_service

CMD [ "python", "./start_opnilog_inference.py" ]
