from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, floor
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier


project_dir = Path(__file__).resolve().parent
input_csv = project_dir / "crime_dataset_india.csv"
output_csv = project_dir / "processed_crime_data.csv"
export_columns = ["Crime Description", "Latitude", "Longitude", "Hour"]

# Ensure Spark workers use the same interpreter as the current venv.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def find_location_column(columns: list[str]) -> str | None:
    keywords = ("state", "ut", "city", "district", "region", "location", "place", "area")
    for name in columns:
        lowered = name.lower()
        if any(keyword in lowered for keyword in keywords):
            return name
    return None


def run_pyspark_pipeline() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("CrimeAnalysis")
        .config("spark.hadoop.hadoop.security.authentication", "simple")
        .config("spark.hadoop.hadoop.security.authorization", "false")
        .config("spark.sql.warehouse.dir", str(project_dir / "spark-warehouse"))
        .getOrCreate()
    )

    try:
        df = spark.read.csv(str(input_csv), header=True, inferSchema=True)
        df = df.dropna()

        location_col = find_location_column(df.columns)
        if location_col:
            df_delhi = df.filter(col(location_col).cast("string").contains("Delhi"))
        else:
            string_cols = [name for name, dtype in df.dtypes if dtype == "string"]
            df_delhi = df
            for name in string_cols:
                candidate = df.filter(col(name).cast("string").contains("Delhi"))
                if candidate.take(1):
                    df_delhi = candidate
                    break

        # Expand dataset 3x.
        df_delhi = df_delhi.union(df_delhi).union(df_delhi)

        # Feature engineering.
        df_delhi = df_delhi.withColumn("Latitude", rand() * (28.9 - 28.4) + 28.4)
        df_delhi = df_delhi.withColumn("Longitude", rand() * (77.5 - 77.0) + 77.0)
        df_delhi = df_delhi.withColumn("Hour", floor(rand() * 24).cast("int"))

        if "Crime Description" not in df_delhi.columns:
            raise KeyError('Required column "Crime Description" not found in dataset')

        top_crimes = (
            df_delhi.groupBy("Crime Description")
            .count()
            .orderBy("count", ascending=False)
        )
        print("Top Crimes:")
        top_crimes.show(5, truncate=False)

        # RandomForest training.
        indexer = StringIndexer(inputCol="Crime Description", outputCol="label")
        df_ml = indexer.fit(df_delhi).transform(df_delhi)

        assembler = VectorAssembler(
            inputCols=["Latitude", "Longitude", "Hour"],
            outputCol="features",
        )
        data = assembler.transform(df_ml).select("features", "label")
        train, test = data.randomSplit([0.8, 0.2], seed=42)

        model = RandomForestClassifier(numTrees=20, seed=42).fit(train)
        preds = model.transform(test)
        print("Sample Predictions:")
        preds.select("prediction", "label").show(5)

        df_delhi.select(*export_columns).toPandas().to_csv(output_csv, index=False)
        print("processed_crime_data.csv created successfully")
    finally:
        spark.stop()


def run_fallback_pipeline() -> None:
    print("PySpark failed. Running fallback...")

    df = pd.read_csv(input_csv).dropna()
    location_col = find_location_column(df.columns.tolist())

    if location_col and location_col in df.columns:
        df_delhi = df[df[location_col].astype(str).str.contains("Delhi", case=False, na=False)].copy()
    else:
        df_delhi = pd.DataFrame()
        for name in df.select_dtypes(include="object").columns.tolist():
            candidate = df[df[name].astype(str).str.contains("Delhi", case=False, na=False)]
            if not candidate.empty:
                df_delhi = candidate.copy()
                break
        if df_delhi.empty:
            df_delhi = df.copy()

    df_delhi = pd.concat([df_delhi, df_delhi, df_delhi], ignore_index=True)

    rng = np.random.default_rng(seed=42)
    df_delhi["Latitude"] = rng.uniform(28.4, 28.9, size=len(df_delhi))
    df_delhi["Longitude"] = rng.uniform(77.0, 77.5, size=len(df_delhi))
    df_delhi["Hour"] = rng.integers(0, 24, size=len(df_delhi))

    if "Crime Description" not in df_delhi.columns:
        raise KeyError('Required column "Crime Description" not found in dataset')

    print("Top Crimes:")
    print(df_delhi["Crime Description"].value_counts().head(5))

    # Keep fallback focused on file creation reliability.
    df_delhi[export_columns].to_csv(output_csv, index=False)
    print("processed_crime_data.csv created successfully")


if __name__ == "__main__":
    try:
        run_pyspark_pipeline()
    except Exception as error:
        print(error)
        run_fallback_pipeline()
