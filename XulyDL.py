#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: data_preprocessing_imputation.py
Mục đích: Xử lý dữ liệu phim từ file CSV. Thay vì loại bỏ hoàn toàn các dòng thiếu giá trị ở các cột quan trọng (trừ release_date),
         ta sẽ thực hiện imputation (điền giá trị) cho các cột số bằng giá trị trung vị.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
import argparse
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cấu hình warnings: chỉ hiển thị những cảnh báo nghiêm trọng (bạn có thể điều chỉnh nếu cần)
warnings.filterwarnings("once", category=UserWarning)

# Các hằng số cấu hình
IRRELEVANT_COLUMNS = [
    "id", "title", "status", "adult", "backdrop_path", "homepage",
    "imdb_id", "original_language", "original_title", "overview",
    "poster_path", "tagline", "production_companies", "production_countries",
    "spoken_languages", "keywords"
]
NUMERIC_COLUMNS = ["revenue", "budget", "vote_average", "vote_count", "runtime", "popularity"]

def load_data(file_path):
    """
    Đọc file CSV và trả về DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info("Dữ liệu được đọc thành công. Số dòng: {}, số cột: {}.".format(df.shape[0], df.shape[1]))
        return df
    except Exception as e:
        logging.error("Lỗi khi đọc file: {}".format(e))
        return None

def drop_irrelevant_columns(df):
    """
    Loại bỏ các cột không cần thiết.
    """
    cols_existing = [col for col in IRRELEVANT_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_existing)
    logging.info("Sau khi loại bỏ các cột không cần thiết, số cột còn lại: {}.".format(df.shape[1]))
    return df

def convert_data_types(df):
    """
    Chuyển đổi kiểu dữ liệu cho các cột:
      - release_date chuyển sang kiểu datetime.
      - Các cột số chuyển sang numeric.
    """
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def handle_duplicates(df):
    """
    Loại bỏ các dòng dữ liệu trùng lặp.
    """
    before_dup = df.shape[0]
    df = df.drop_duplicates()
    logging.info("Đã loại bỏ {} dòng dữ liệu trùng lặp.".format(before_dup - df.shape[0]))
    return df

def handle_missing_values_imputation(df):
    """
    Xử lý các giá trị thiếu:
      - Với các cột số: điền giá trị thiếu bằng giá trị trung vị.
      - Với cột release_date: loại bỏ các dòng thiếu (imputation cho ngày không hợp lý).
      - Với cột genres: điền giá trị thiếu bằng chuỗi rỗng.
    """
    # Imputation cho các cột số bằng giá trị trung vị
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_val = df[col].median()
                logging.info("Cột '{}' có {} giá trị thiếu. Điền giá trị trung vị: {}.".format(col, missing_count, median_val))
                df[col] = df[col].fillna(median_val)
    
    # Xử lý cột release_date: loại bỏ các dòng thiếu
    if "release_date" in df.columns:
        missing_release = df["release_date"].isnull().sum()
        if missing_release > 0:
            logging.info("Cột 'release_date' có {} giá trị thiếu. Loại bỏ các dòng này.".format(missing_release))
        before = df.shape[0]
        df = df.dropna(subset=["release_date"])
        after = df.shape[0]
        logging.info("Đã loại bỏ {} dòng do thiếu release_date.".format(before - after))
    
    # Với genres, điền giá trị thiếu bằng chuỗi rỗng
    if "genres" in df.columns:
        missing_genres = df["genres"].isnull().sum()
        if missing_genres > 0:
            logging.info("Cột 'genres' có {} giá trị thiếu. Điền chuỗi rỗng.".format(missing_genres))
        df["genres"] = df["genres"].fillna("")
    
    return df

def feature_engineering(df):
    """
    Tạo thêm các đặc trưng mới:
      - Tạo cột 'year' và 'month' từ 'release_date'.
      - Tạo cột 'profit' = revenue - budget.
      - Xử lý cột 'genres': tách thành danh sách và one-hot encoding.
    """
    if "release_date" in df.columns:
        df["year"] = df["release_date"].dt.year
        df["month"] = df["release_date"].dt.month
    
    if "revenue" in df.columns and "budget" in df.columns:
        df["profit"] = df["revenue"] - df["budget"]
    
    # Xử lý cột genres: tách chuỗi thành danh sách và one-hot encoding
    if "genres" in df.columns:
        df["genres_list"] = df["genres"].apply(lambda x: [genre.strip() for genre in x.split(",") if genre.strip()])
        mlb = MultiLabelBinarizer()
        genres_dummies = pd.DataFrame(
            mlb.fit_transform(df["genres_list"]),
            columns=["genre_" + g for g in mlb.classes_],
            index=df.index
        )
        df = pd.concat([df, genres_dummies], axis=1)
        # Loại bỏ cột gốc và cột trung gian
        df = df.drop(columns=["genres", "genres_list"])
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Xử lý dữ liệu phim từ file CSV")
    parser.add_argument("--input_file", type=str, default="TMDB_movie_dataset_v11.csv", help="Đường dẫn tới file CSV đầu vào")
    parser.add_argument("--output_file", type=str, default="movies_cleaned.csv", help="Đường dẫn tới file CSV đầu ra")
    args = parser.parse_args()
    
    # Bước 1: Đọc dữ liệu
    df = load_data(args.input_file)
    if df is None:
        return
    
    # Bước 2: Loại bỏ các cột không liên quan
    df = drop_irrelevant_columns(df)
    
    # Bước 3: Chuyển đổi kiểu dữ liệu
    df = convert_data_types(df)
    
    # Bước 4: Xử lý dữ liệu trùng lặp
    df = handle_duplicates(df)
    
    # Bước 5: Xử lý giá trị thiếu (imputation cho các cột số và loại bỏ các dòng thiếu release_date)
    df = handle_missing_values_imputation(df)
    
    # Bước 6: Tạo các đặc trưng mới
    df = feature_engineering(df)
    
    # In thông tin DataFrame sau xử lý (có thể sử dụng logging hoặc print)
    logging.info("\nThông tin DataFrame sau xử lý:")
    logging.info(df.info())
    logging.info("\n5 dòng dữ liệu đầu:")
    logging.info(df.head())
    
    # Lưu dữ liệu đã xử lý vào file CSV mới
    df.to_csv(args.output_file, index=False)
    logging.info("\nDữ liệu đã được lưu vào file '{}'.".format(args.output_file))

if __name__ == '__main__':
    main()
