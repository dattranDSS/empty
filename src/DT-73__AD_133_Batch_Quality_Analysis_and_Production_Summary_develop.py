spark.catalog.setCurrentCatalog("automation_playground")

# PySpark script for Databricks - Batch QC Analysis and Reporting
# Purpose: Evaluate batch QC results, assign pass/fail, and compute/report product-wise and summary analytics.
# Author: Dat Tran
# Date: 2025-07-21
# Description: Reads 'automation_playground.purgo_playground.batch_qc', standardizes and processes batch QC data,
#              assigns quality status, computes product-wise and total production analytics,
#              persists outputs as per acceptance criteria, with robust error handling and schema/data validation.

# Commented out SparkSession initialization (Databricks provides 'spark')
from pyspark.sql import SparkSession  # Builtin pyspark

# ======================== IMPORTS ========================
from pyspark.sql import functions as F         
from pyspark.sql.window import Window         
from pyspark.sql.types import (StructType,    
                              StructField,
                              StringType,
                              DoubleType,
                              LongType)

# ======================== CONFIGURATION ========================
SOURCE_QC_TABLE = "automation_playground.purgo_playground.batch_qc"
BATCHWISE_TARGET = "automation_playground.purgo_playground.batch_summary"
PRODUCTWISE_TARGET = "automation_playground.purgo_playground.product_wise_batch_analysis"
SUMMARY_TARGET = "automation_playground.purgo_playground.batch_production_summary"

# ======================== HELPER FUNCTIONS ========================
def safe_read_table(table_path: str):
    """
    Safely reads a table from Unity Catalog.
    Args:
        table_path (str): catalog.schema.table path
    Returns:
        DataFrame: Loaded table as DataFrame
    Raises:
        RuntimeError: If read fails
    """
    try:
        return spark.read.table(table_path)
    except Exception as e:
        raise RuntimeError(f"Error reading table {table_path}: {e}")

def safe_save_table(df, table_path: str):
    """
    Safely writes DataFrame to a Delta table with overwrite mode.
    Args:
        df (DataFrame): The DataFrame to save
        table_path (str): catalog.schema.table path
    Raises:
        RuntimeError: If write fails
    """
    try:
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_path)
    except Exception as e:
        raise RuntimeError(f"Error saving table {table_path}: {e}")

def is_blank_or_null(col):
    """
    Helper for blank or null string detection.
    Args:
        col (Column): Spark Column
    Returns:
        Column: Boolean column
    """
    return F.col(col).isNull() | (F.trim(F.col(col)) == "")

# ======================== DATA EXTRACTION AND FILTERING ========================
# -- Load batch_qc, exclude rows with null/empty batch_id/product_name or null score
batch_qc_df = safe_read_table(SOURCE_QC_TABLE)
filtered_qc_df = batch_qc_df \
    .where(~is_blank_or_null("batch_id")) \
    .where(~is_blank_or_null("product_name")) \
    .where(F.col("quality_check_score").isNotNull())

# ======================== DATA STANDARDIZATION AND QUALITY STATUS ========================
def assign_quality_status(df):
    """
    Assigns quality_status ('pass' if score >=97 else 'fail'), and standardizes product_name (trimmed, lowercase).
    Args:
        df (DataFrame): Batch QC DataFrame
    Returns:
        DataFrame: batch_id, standardized product_name, quality_check_score, quality_status
    """
    return df.select(
        F.col("batch_id"),
        F.lower(F.trim(F.col("product_name"))).alias("product_name"),
        F.col("quality_check_score")
    ).withColumn(
        "quality_status",
        F.when(F.col("quality_check_score") >= 97.0, F.lit("pass")).otherwise(F.lit("fail"))
    )

batch_status_df = assign_quality_status(filtered_qc_df)

# ======================== PRODUCT-WISE ANALYSIS ========================
def compute_productwise_metrics(df):
    """
    Computes product-level total, pass, fail, and % pass for batches.
    Args:
        df (DataFrame): DataFrame with standardized product_name and quality_status
    Returns:
        DataFrame: product_name, total_batches, passed_batches, failed_batches, percentage_passed_batches
    """
    summary = df.groupBy("product_name").agg(
        F.count("*").alias("total_batches"),
        F.sum(F.when(F.col("quality_status") == "pass", 1).otherwise(0)).cast(LongType()).alias("passed_batches"),
        F.sum(F.when(F.col("quality_status") == "fail", 1).otherwise(0)).cast(LongType()).alias("failed_batches")
    ).withColumn(
        "percentage_passed_batches",
        F.round(F.col("passed_batches") * 100.0 / F.col("total_batches"), 2)
    )
    return summary

productwise_df = compute_productwise_metrics(batch_status_df)

# ======================== BATCH-WISE PRODUCTION SUMMARY ========================
def compute_total_batch_summary(df):
    """
    Produces total batches, total passed, and total failed (single row).
    Args:
        df (DataFrame): DataFrame with quality_status
    Returns:
        DataFrame: total_batches, total_passed_batches, total_failed_batches (single row)
    """
    return df.agg(
        F.count("*").alias("total_batches"),
        F.sum(F.when(F.col("quality_status") == "pass", 1).otherwise(0)).cast(LongType()).alias("total_passed_batches"),
        F.sum(F.when(F.col("quality_status") == "fail", 1).otherwise(0)).cast(LongType()).alias("total_failed_batches"),
    )

batch_summary_df = compute_total_batch_summary(batch_status_df)

# ======================== DATA VALIDATION ========================
# -- Validate required columns/schema counts to target tables
expected_batchwise_cols = ["batch_id", "product_name", "quality_check_score", "quality_status"]
expected_productwise_cols = [
    "product_name", "total_batches", "passed_batches", "failed_batches", "percentage_passed_batches"
]
expected_summary_cols = [
    "total_batches", "total_passed_batches", "total_failed_batches"
]
assert batch_status_df.columns == expected_batchwise_cols, f"Batch output columns mismatch: {batch_status_df.columns}"
assert all(c in productwise_df.columns for c in expected_productwise_cols), "Productwise columns mismatch"
assert all(c in batch_summary_df.columns for c in expected_summary_cols), "Summary columns mismatch"

# ======================== PERSIST OUTPUT TO DELTA TABLES ========================
# -- Write batch-wise status
safe_save_table(batch_status_df, BATCHWISE_TARGET)
# -- Write product-wise statistics
safe_save_table(productwise_df, PRODUCTWISE_TARGET)
# -- Write overall batch production summary
safe_save_table(batch_summary_df, SUMMARY_TARGET)

# ======================== END OF BATCH QC ANALYSIS SCRIPT ========================
