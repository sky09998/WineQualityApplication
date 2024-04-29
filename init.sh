#!/bin/bash
pip install pyspark numpy boto3
aws s3api get-object --bucket winequalityapplication --key WineQualityTrainingAndPrediction.py /home/hadoop/WineQualityTrainingAndPrediction.py
curl -o hadoop-aws-3.0.0.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.0.0/hadoop-aws-3.0.0.jar
curl -o aws-java-sdk-1.11.375.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.11.375/aws-java-sdk-1.11.375.jar