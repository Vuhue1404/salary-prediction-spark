from pyspark.sql import SparkSession, Row
from pyspark.ml import Pipeline
from pyspark.sql.functions import lit
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# ======================================================
# Build Pipeline
# ======================================================
def build_pipeline():
    categorical_cols = [
        'experience_level', 'employment_type', 'job_title',
        'employee_residence', 'company_location', 'company_size'
    ]

    stages = []

    # Encode categorical features
    for c in categorical_cols:
        stages.append(StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep"
        ))
        stages.append(OneHotEncoder(
            inputCol=f"{c}_idx",
            outputCol=f"{c}_ohe"
        ))

    # Scale numeric features
    num_assembler = VectorAssembler(
        inputCols=['work_year', 'remote_ratio'],
        outputCol="num_features"
    )
    scaler = StandardScaler(
        inputCol="num_features",
        outputCol="num_scaled"
    )
    stages += [num_assembler, scaler]

    # Assemble all features
    assembler_inputs = [f"{c}_ohe" for c in categorical_cols] + ["num_scaled"]
    stages.append(VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features"
    ))

    # Random Forest model
    stages.append(RandomForestRegressor(
        featuresCol="features",
        labelCol="salary_in_usd",
        numTrees=100,
        seed=42
    ))

    return Pipeline(stages=stages)


# ======================================================
# Train & Save Model
# ======================================================
def train_and_save_model(df):
    pipeline = build_pipeline()

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_df)

    predictions = model.transform(test_df)

    r2 = RegressionEvaluator(
        labelCol="salary_in_usd",
        predictionCol="prediction",
        metricName="r2"
    ).evaluate(predictions)

    print(f" Model trained successfully | R2 Score: {r2:.4f}")

    model.write().overwrite().save("output/spark_rf_model")
    print(" Model saved to output/spark_rf_model")


# ======================================================
# Evaluate Model Multiple Runs
# ======================================================
def evaluate_model_multiple_runs(df, n_runs=10):
    pipeline = build_pipeline()
    df.cache()

    rmse_eval = RegressionEvaluator(
        labelCol="salary_in_usd",
        predictionCol="prediction",
        metricName="rmse"
    )
    mae_eval = RegressionEvaluator(
        labelCol="salary_in_usd",
        predictionCol="prediction",
        metricName="mae"
    )
    r2_eval = RegressionEvaluator(
        labelCol="salary_in_usd",
        predictionCol="prediction",
        metricName="r2"
    )

    results = []

    for i in range(n_runs):
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42 + i)

        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        rmse = rmse_eval.evaluate(predictions)
        mae = mae_eval.evaluate(predictions)
        r2 = r2_eval.evaluate(predictions)
        results.append(Row(
            Run=i + 1,
            RMSE=round(rmse, 2),
            MAE=round(mae, 2),
            R2=round(r2, 4)
        ))
    results_df = spark.createDataFrame(results)

    avg_df = results_df.selectExpr(
        "avg(RMSE) as RMSE",
        "avg(MAE) as MAE",
        "avg(R2) as R2"
    ).withColumn("Run", lit("AVERAGE")).select("Run", "RMSE", "MAE", "R2")
    final_df = results_df.unionByName(avg_df)
    print("\n BẢNG ĐÁNH GIÁ MÔ HÌNH (NHIỀU LẦN CHẠY)")
    final_df.orderBy("Run").show(truncate=False)

    # Unpersist dữ liệu
    df.unpersist()

    return final_df

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    spark = SparkSession.builder .appName("Salary_Model_RandomForest").getOrCreate()

    # Load preprocessed data
    df = spark.read.parquet("output/preprocessed_data")

    # Train & save model
    train_and_save_model(df)

    # Evaluate model 10 runs
    evaluate_model_multiple_runs(df, n_runs=10)

    spark.stop()
