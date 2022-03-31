FROM rancher/opni-python-base:3.8-torch

RUN zypper -n ref  && \
    zypper --non-interactive in wget && \
    zypper --non-interactive in zip

COPY ./nulog-inference-service/ /app/
COPY ./models/nulog/ /app/

RUN chmod a+rwx -R /app
WORKDIR /app

CMD [ "python", "start-nulog-inference.py" ]
