import sys
import shutil
import os
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


access_key = os.getenv("ACCESSKey")
secret_key = os.getenv("SECRETKey")

# Initialize S3 client
s3_client = boto3.client('s3')

def download_directory_from_s3(bucket_name, s3_folder, local_dir):
    """Download an entire directory from an S3 bucket to a local path."""
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get('Contents', []):
            # Define the local path to save the file
            local_file_path = os.path.join(local_dir, obj['Key'][len(s3_folder):])
            local_file_dir = os.path.dirname(local_file_path)

            # Ensure the local directory exists
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            # Download the file
            s3.download_file(bucket_name, obj['Key'], local_file_path)
            print(f"Downloaded {obj['Key']} to {local_file_path}")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols, num_but_cat, cat_but_car = [], [], []
    for field in dataframe.schema.fields:
        distinct_count = dataframe.select(field.name).distinct().count()
        if str(field.dataType) == 'StringType':
            if distinct_count > car_th:
                cat_but_car.append(field.name)
            else:
                cat_cols.append(field.name)
        elif distinct_count < cat_th:
            num_but_cat.append(field.name)

    cat_cols = list(set(cat_cols) - set(cat_but_car))
    num_cols = [f.name for f in dataframe.schema.fields if str(f.dataType) != 'StringType' and f.name not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

def get_decision_tree_params(labelCol):
    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol=labelCol)
    dt = DecisionTreeClassifier(featuresCol="scaledFeatures", labelCol=labelCol)

    return [
        ("LR", lr, ParamGridBuilder()
             .addGrid(lr.maxIter, [10, 20, 50])
             .addGrid(lr.regParam, [0.01, 0.1, 0.5])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build()),
        ("DT", dt, ParamGridBuilder()
             .addGrid(dt.maxDepth, [3, 5, 10])
             .addGrid(dt.maxBins, [20, 40, 60])
             .addGrid(dt.impurity, ["entropy", "gini"])
             .build())
    ]

def evaluate_models(train_data, valid_data, featuresCol, labelCol):
    assembler = VectorAssembler(inputCols=featuresCol, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, metricName="f1")
    best_f1_score, best_model = 0, None

    for name, model, paramGrid in get_decision_tree_params(labelCol):
        pipeline = Pipeline(stages=[assembler, scaler, model])
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
        cv_model = cv.fit(train_data)
        f1_score = evaluator.evaluate(cv_model.transform(valid_data))
        if f1_score > best_f1_score:
            best_f1_score, best_model = f1_score, cv_model.bestModel
            print(f"{name} - Best F1 Score: {f1_score:.2f}")

    return best_model

def fetch_dataframe_from_s3(key, spark, data_transformations):
    response = s3_client.get_object(Bucket='winequalityapplication', Key=key)
    data_string = response['Body'].read().decode('utf-8').replace('"', '')
    data_list = [tuple(x.split(';')) for x in data_string.strip().split('\r\n') if x]
    columns = list(data_list.pop(0))
    df = spark.createDataFrame(data_list, columns)
    return data_transformations(df)

def data_transformations(df):
    float_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                  "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                  "pH", "sulphates", "alcohol"]
    for col in float_cols:
        df = df.withColumn(col, df[col].cast(FloatType()))
    df = df.withColumn('quality', df['quality'].cast(IntegerType()))
    return df

def predict_new_data(new_data_path, spark, best_model):
    new_df = fetch_dataframe_from_s3(new_data_path, spark, data_transformations)
    temp_quality_column_data = new_df.select("quality")
    new_df = new_df.drop("quality")
    predictions = best_model.transform(new_df)
    predictions.show()  # Display some of the predictions
    predictions_with_column = predictions.join(temp_quality_column_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1Score = evaluator.evaluate(predictions_with_column)
    print(f"f1Score {f1Score:.2f}")
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions_with_column)
    print(f"accuracy {accuracy:.2f}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Wine Quality Prediction") \
        .config("spark.jars", "hadoop-aws-3.0.0.jar,aws-java-sdk-1.11.375.jar") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem").getOrCreate()

    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", f"{access_key}")
    spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", f"{secret_key}")
    train_df = fetch_dataframe_from_s3('TrainingDataset.csv', spark, data_transformations)
    valid_df = fetch_dataframe_from_s3('ValidationDataset.csv', spark, data_transformations)
    cat_cols, num_cols, _ = grab_col_names(train_df)

    featuresCol = cat_cols + num_cols
    if 'quality' in featuresCol:
        featuresCol.remove('quality')

    if '--train' in sys.argv:
        best_model = evaluate_models(train_df, valid_df, featuresCol, 'quality')
        # Save best model to S3
        model_path = "s3://winequalityapplication/bestmodel"
        best_model.write().overwrite().save(model_path)

    if '--predict' in sys.argv:
        download_directory_from_s3('winequalityapplication','bestmodel/','/home/hadoop/bestmodel')
        best_model = PipelineModel.load('/home/hadoop/bestmodel')
        predict_new_data('TestDataset.csv', spark, best_model)

    spark.stop()
