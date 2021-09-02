FROM rancher/opni-python-base:3.8-torch

RUN apt-get update \
    && apt-get install -y wget zip \
    && apt-get clean

COPY ./nulog-inference-service/ /app/
COPY ./models/nulog/ /app/

RUN chmod a+rwx -R /app
WORKDIR /app

CMD [ "python", "start-nulog-inference.py" ]
