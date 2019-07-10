#FROM openjdk:jre


FROM frolvlad/alpine-java:jdk8-slim

RUN apk add --update \
    python3 \
    python3-dev \
    py-pip \
    build-base \
    openjdk8-jre \
  && pip install virtualenv \
  && rm -rf /var/cache/apk/*
RUN pip install --upgrade pip


RUN pip3 install mlflow

ADD target/LSTM-1.0-SNAPSHOT.jar /opt/random-generator.jar
EXPOSE 5000
# CMD java -jar /opt/random-generator.jar


