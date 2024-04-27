FROM amazonlinux:latest
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8
ENV JAVA_HOME /usr/local/openjdk-8
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1
WORKDIR /app
COPY WineQualityTrainingAndPrediction.py /app
COPY CleanTrainingDataset.csv /app
COPY CleanValidationDataset.csv /app
COPY entrypoint.sh /app
RUN yum update
RUN yum install python -y
RUN yum install python-pip -y
ENTRYPOINT [entrypoint.sh]