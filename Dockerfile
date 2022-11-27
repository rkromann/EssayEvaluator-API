FROM python:3.9

RUN useradd -ms /bin/bash api

WORKDIR /api
EXPOSE 8000

RUN mkdir -p ./artifacts
RUN chown -R api:api ./artifacts

USER api

COPY ./requirements.txt /api/requirements.txt
RUN pip install -r ./requirements.txt --default-timeout=100 future

COPY ./config.ini /api/config.ini

COPY ./api /api
COPY ./src /api/src

ENV PATH="/home/api/.local/bin:${PATH}"

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]