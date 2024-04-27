FROM amazonlinux:latest
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8
ENV JAVA_HOME /usr/local/openjdk-8
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1
WORKDIR /app
COPY winequalityapp.py /app
COPY best_model /app/best_model
COPY entrypoint.sh /app
RUN yum update
RUN yum install python -y
RUN yum install python-pip -y
RUN pip install setuptools
RUN pip install pyspark
RUN pip install numpy
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
