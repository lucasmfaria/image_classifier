FROM python:3.9-slim-buster

WORKDIR /opt
COPY . .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN apt update && apt install wget -y \
    && mkdir -p /root/.keras/models \
    && wget https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /root/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 && apt install poppler-utils -y
CMD python