# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import (
#     StringIndexer,
#     OneHotEncoder,
#     VectorAssembler,
#     StandardScaler
# )
#
# # =========================
# # Spark Session
# # =========================
# spark = SparkSession.builder \
#     .appName("IT Salary Preprocess") \
#     .getOrCreate()
#
# # =========================
# # Load data
# # =========================
# df = spark.read.csv(
#     "..\data\Data_salary.csv",
#     header=True,
#     inferSchema=True
# )
#
# # =========================
# # Clean data
# # =========================
# keep_columns = [
#     'work_year',
#     'experience_level',
#     'employment_type',
#     'job_title',
#     'salary_in_usd',
#     'employee_residence',
#     'remote_ratio',
#     'company_location',
#     'company_size'
# ]
#
# df = df.select(*keep_columns)
# df = df.dropDuplicates()
# df = df.filter(col("salary_in_usd") <= 350000)
#
# # =========================
# # Categorical columns
# # =========================
# categorical_cols = [
#     'experience_level',
#     'employment_type',
#     'job_title',
#     'employee_residence',
#     'company_location',
#     'company_size'
# ]
#
# indexers = [
#     StringIndexer(
#         inputCol=c,
#         outputCol=f"{c}_idx",
#         handleInvalid="keep"
#     )
#     for c in categorical_cols
# ]
#
# encoders = [
#     OneHotEncoder(
#         inputCol=f"{c}_idx",
#         outputCol=f"{c}_ohe"
#     )
#     for c in categorical_cols
# ]
#
# # =========================
# # Numeric columns
# # =========================
# numeric_cols = ['work_year', 'remote_ratio']
#
# num_assembler = VectorAssembler(
#     inputCols=numeric_cols,
#     outputCol="numeric_vec"
# )
#
# scaler = StandardScaler(
#     inputCol="numeric_vec",
#     outputCol="numeric_scaled",
#     withMean=True,
#     withStd=True
# )
#
# # =========================
# # Assemble final features
# # =========================
# final_assembler = VectorAssembler(
#     inputCols=["numeric_scaled"] + [f"{c}_ohe" for c in categorical_cols],
#     outputCol="features"
# )
#
# pipeline = Pipeline(stages=[
#     *indexers,
#     *encoders,
#     num_assembler,
#     scaler,
#     final_assembler
# ])
#
# model = pipeline.fit(df)
# df_processed = model.transform(df)
#
# # =========================
# # Save ONLY needed columns
# # =========================
# df_processed.select(
#     "features",
#     "salary_in_usd"
# ).write.mode("overwrite").parquet("output/preprocessed_parquet")
#
# print("✅ Preprocessing completed (Spark)")
# spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def clean_data_spark():
    spark = SparkSession.builder.appName("Preprocess_Salary").getOrCreate()

    # 1. Load data
    df = spark.read.csv("../data/Data_salary.csv", header=True, inferSchema=True)

    # 2. Drop duplicates
    df = df.dropDuplicates()

    # 3. Select columns & Filter outliers
    keep_columns = [
        'work_year', 'experience_level', 'employment_type', 'job_title',
        'salary_in_usd', 'employee_residence', 'remote_ratio',
        'company_location', 'company_size'
    ]
    df = df.select(*keep_columns)
    df = df.filter(col("salary_in_usd") <= 350000)

    # 4. Save cleaned data
    output_path = "output/preprocessed_data"
    df.write.mode("overwrite").parquet(output_path)

    print(f"✅ Data cleaned and saved to: {output_path}")
    spark.stop()


if __name__ == "__main__":
    clean_data_spark()