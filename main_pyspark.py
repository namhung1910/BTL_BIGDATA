# -*- coding: utf-8 -*-
"""
Pipeline dự đoán 'trend' với Spark MLlib:
    - Tính median của profit và tạo biến target 'trend'
    - Xử lý cột genre: gộp các cột hiếm thành 'genre_rare' (theo ngưỡng điều chỉnh)
    - Oversampling lớp minority (trend = 1)
    - Huấn luyện Random Forest và đánh giá qua:
         + Confusion Matrix
         + ROC Curve
         + Feature Importance
    - EDA: Vẽ 3 biểu đồ khám phá dữ liệu gốc
"""

import sys
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, sum as spark_sum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# -------------------- PHẦN 1: XỬ LÝ DỮ LIỆU VỚI SPARK --------------------
spark = SparkSession.builder.appName("MovieTrendPrediction_Optimized").getOrCreate()

# Đọc dữ liệu CSV
data_path = r"C:/Users/LENOVO/Documents/BTL_BigData/movies_cleaned.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
print("=== 5 dòng đầu dữ liệu ===")
df.show(5)

# Tạo biến target 'trend' dựa trên median của profit
median_profit = df.approxQuantile("profit", [0.5], 0.01)[0]
print(f"Median profit: {median_profit}")
df = df.withColumn("trend", when(col("profit") >= median_profit, 1).otherwise(0))

# Loại bỏ các cột không cần thiết (tránh rò rỉ thông tin)
drop_cols = ['release_date', 'revenue', 'budget', 'profit', 'popularity', 'month', 'year']
df = df.drop(*drop_cols)
print("=== Các cột còn lại:", df.columns)

# Xác định danh sách feature (ngoại trừ target 'trend')
feature_cols = [c for c in df.columns if c != "trend"]
print("=== Features ban đầu:", feature_cols)

# Xử lý cột genre: gom các cột genre hiếm (theo ngưỡng ratio < 0.05)
genre_cols = [c for c in feature_cols if c.startswith("genre_")]
total = df.count()
genre_threshold = 0.05
sparse_genres = [
    g for g in genre_cols 
    if df.agg(spark_sum(col(g)).alias("ones")).first()["ones"] / total < genre_threshold
]
print(f"=== Các cột genre hiếm (ratio < {genre_threshold}):", sparse_genres)

if sparse_genres:
    df = df.withColumn(
        "genre_rare", 
        (reduce(lambda a, b: a + b, [col(g) for g in sparse_genres]) > 0).cast("integer")
    )
    feature_cols = [c for c in feature_cols if c not in sparse_genres] + ["genre_rare"]
    print("Đã tạo 'genre_rare'. Feature cập nhật:", feature_cols)
else:
    print("Không có genre nào quá hiếm.")

# Tạo vector features và chọn cột cần thiết cho mô hình
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_model = assembler.transform(df).select("features", "trend")
df_model.printSchema()

# Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)
print(f"Tập huấn luyện: {train_df.count()} dòng, Tập test: {test_df.count()} dòng")

# -------------------- OVERSAMPLING LỚP MINORITY --------------------
train_majority = train_df.filter(col("trend") == 0)
train_minority = train_df.filter(col("trend") == 1)
count_majority = train_majority.count()
count_minority = train_minority.count()
print(f"Majority: {count_majority}, Minority: {count_minority}")

if count_minority == 0:
    print("Không có mẫu minority, kết thúc chương trình.")
    spark.stop()
    sys.exit(1)

ratio = count_majority / count_minority
oversampled_minority = train_minority.sample(withReplacement=True, fraction=ratio, seed=42)
train_df_balanced = train_majority.union(oversampled_minority)
print(f"Sau oversampling: {train_df_balanced.count()} dòng "
      f"(Majority: {train_df_balanced.filter(col('trend')==0).count()}, "
      f"Minority: {train_df_balanced.filter(col('trend')==1).count()})")

# -------------------- HUẤN LUYỆN VÀ ĐÁNH GIÁ --------------------
rf = RandomForestClassifier(labelCol="trend", featuresCol="features", numTrees=100, seed=42)
model = rf.fit(train_df_balanced)
print("Random Forest đã được huấn luyện.")
predictions = model.transform(test_df)
predictions.select("trend", "prediction", "probability").show(5)
accuracy = MulticlassClassificationEvaluator(
    labelCol="trend", predictionCol="prediction", metricName="accuracy"
).evaluate(predictions)
print(f"Độ chính xác (Accuracy): {accuracy:.4f}")

# -------------------- PHẦN 2: VẼ BIỂU ĐỒ VỚI MATPLOTLIB/SEABORN --------------------
# Chuyển kết quả dự đoán sang Pandas
pred_pd = predictions.select("trend", "prediction", "probability").toPandas()
pred_pd["prob_class1"] = pred_pd["probability"].apply(lambda x: x[1])

# 1. Confusion Matrix & Classification Report
cm = confusion_matrix(pred_pd["trend"], pred_pd["prediction"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Confusion Matrix")
plt.show()

print("=== Classification Report ===")
print(classification_report(pred_pd["trend"], pred_pd["prediction"], zero_division=0))

# 2. ROC Curve & AUC
fpr, tpr, _ = roc_curve(pred_pd["trend"], pred_pd["prob_class1"])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 3. Feature Importance
feat_imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.featureImportances.toArray()
}).sort_values("importance", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feat_imp_df)
plt.title("Feature Importance")
plt.xlabel("Tầm quan trọng")
plt.ylabel("Feature")
plt.show()

# -------------------- PHẦN 3: EDA - VẼ 3 BIỂU ĐỒ KHÁM PHÁ DỮ LIỆU --------------------
# Lấy mẫu 1% dữ liệu gốc để chuyển sang Pandas (tránh vấn đề bộ nhớ)
data_pd = df.sample(withReplacement=False, fraction=0.01, seed=42).toPandas()

# Biểu đồ phân phối vote_average
plt.figure(figsize=(8, 6))
sns.histplot(data_pd["vote_average"], bins=30, kde=True, color="skyblue")
plt.title("Phân phối vote_average")
plt.xlabel("vote_average")
plt.ylabel("Số lượng phim")
plt.show()

# Biểu đồ số lượng phim theo thể loại
genre_columns = [c for c in data_pd.columns if c.startswith("genre_")]
genre_counts = data_pd[genre_columns].sum().reset_index()
genre_counts.columns = ["genre", "count"]
plt.figure(figsize=(12, 6))
sns.barplot(x="genre", y="count", data=genre_counts, palette="viridis")
plt.title("Số lượng phim theo các thể loại")
plt.xlabel("Thể loại")
plt.ylabel("Số lượng phim")
plt.xticks(rotation=45)
plt.show()

# Biểu đồ phân tán giữa vote_count và vote_average
plt.figure(figsize=(10, 6))
sns.scatterplot(x="vote_count", y="vote_average", data=data_pd, alpha=0.6)
plt.title("Mối quan hệ giữa vote_count và vote_average")
plt.xlabel("vote_count")
plt.ylabel("vote_average")
plt.show()

spark.stop()
