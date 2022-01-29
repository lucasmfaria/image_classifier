FROM python:3.9-slim-buster

WORKDIR /opt
COPY . .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
CMD python