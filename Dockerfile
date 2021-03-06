FROM rancher/opni-python-base:3.8-torch

RUN zypper -n ref  && \
    zypper --non-interactive in wget && \
    zypper --non-interactive in zip

COPY ./opnilog-inference-service/ /app/
COPY ./models/opnilog/ /app/

RUN chmod a+rwx -R /app
RUN pip install --no-cache-dir -r /app/requirements.txt
WORKDIR /app

CMD [ "python", "start_opnilog_inference.py" ]
