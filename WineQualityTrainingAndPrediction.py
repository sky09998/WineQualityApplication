import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import boto3

s3_client = boto3.client('s3')

def get_data_from_s3(bucket_name, file_key):
    s3_client.get_object(bucket_name, file_key)

def upload_model_to_s3(bucket_name, local_model_path, s3_model_key):
    s3_client.upload_file(local_model_path, bucket_name, s3_model_key)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols, num_but_cat, cat_but_car = [], [], []
    for field in dataframe.schema.fields:
        if str(field.dataType) == 'StringType':
            if dataframe.select(field.name).distinct().count() > car_th:
                cat_but_car.append(field.name)
            else:
                cat_cols.append(field.name)
        else:
            if dataframe.select(field.name).distinct().count() < cat_th:
                num_but_cat.append(field.name)

    cat_cols = list(set(cat_cols + num_but_cat) - set(cat_but_car))
    num_cols = [field.name for field in dataframe.schema.fields if str(field.dataType) != 'StringType' and field.name not in num_but_cat]

    print(f"Observations: {dataframe.count()}, Variables: {len(dataframe.columns)}")
    print(f'cat_cols: {len(cat_cols)}, num_cols: {len(num_cols)}, cat_but_car: {len(cat_but_car)}, num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def get_models(labelCol):
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

def evaluate_models(training_data, validation_data, featuresCol, labelCol):
    featureIndexer = VectorAssembler(inputCols=featuresCol, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    best_f1_score, best_cv_model, best_model_name = 0, None, ""
    evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="f1")

    for name, model, paramGrid in get_models(labelCol):
        pipeline = Pipeline(stages=[featureIndexer, scaler, model])
        cv = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=5)  # Number of folds for cross-validation
        cv_model = cv.fit(training_data)
        predictions = cv_model.transform(validation_data)
        f1_score = evaluator.evaluate(predictions)

        print(f"{name} - Best F1 Score: {f1_score:.2f}")

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model_name = name
            best_cv_model = cv_model.bestModel

    if best_cv_model:
        # best_cv_model.write().overwrite().save("s3a://winequalityapplication/model/best_model")
        print(f"Best Model: {best_model_name} with F1 Score: {best_f1_score:.2f}")

    return best_cv_model

def predict_new_data(new_data_path):
    spark = SparkSession.builder.appName("Prediction Using Best Model").getOrCreate()
    new_data = spark.read.csv(get_data_from_s3(new_data_s3_path, "Test"), header=True, inferSchema=True)
    temp_quality_column_data = new_data.select("quality")
    new_data = new_data.drop("quality")
    best_model = PipelineModel.load("s3a://winequalityapplication/best_model")
    predictions = best_model.transform(new_data)
    predictions.select("prediction").show()
    predictions_with_column = predictions.join(temp_quality_column_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1Score = evaluator.evaluate(predictions_with_column)
    print("f1Score ",f1Score)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions_with_column)
    print("accuracy ",accuracy)
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 5:  # Adjusted for S3 file paths
        print("Usage: spark-submit script.py <TrainingDataSet S3 path> <ValidationDataSet S3 path> <NewDataSet S3 path> <S3 Model Bucket>")
        sys.exit(-1)

    spark = SparkSession.builder.appName("Model Training and Validation").getOrCreate()
    training_data_s3_path, validation_data_s3_path, new_data_s3_path, model_bucket = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # Read datasets from local file system
    training_data = spark.read.csv(get_data_from_s3(training_data_s3_path, "CleanTrainingDataset.csv"), header=True, inferSchema=True)
    validation_data = spark.read.csv(get_data_from_s3(validation_data_s3_path, "CleanValidationDataset.csv"), header=True, inferSchema=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(training_data)
    featuresCol = cat_cols + num_cols
    featuresCol = [col for col in featuresCol if col != 'quality']
    labelCol = 'quality'  # Update this as per your dataset

    if '--train' in sys.argv:
        best_model = evaluate_models(training_data, validation_data, featuresCol, labelCol)
        # Save best model to local file system
        local_model_path = '/best_model'
        best_model.write().overwrite().save(local_model_path)
        # Upload best model to S3
        upload_model_to_s3(model_bucket, local_model_path, 'best_model')

    if '--predict' in sys.argv:
        # Download new data from S3 to local file system
        predict_new_data(new_data_s3_path)

    spark.stop()