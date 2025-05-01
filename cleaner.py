import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pathlib import Path
import warnings

def clean_data(timestamp, input_file, id_columns=None, outlier_contamination=0.05, remove_outliers=False,
               exclude_from_imputation=None, skip_outlier_file=False, skip_missing_rows_file=False):
    """
    Clean the dataset by handling missing values and outliers.

    Args:
        timestamp: Timestamp of the dataset analysis
        input_file (str): Path to the input CSV file
        id_columns (list, optional): List of columns to treat as identifiers
        outlier_contamination (float, optional): Expected proportion of outliers in the dataset (0-0.5)
        remove_outliers (bool, optional): Whether to remove outliers from the dataset
        exclude_from_imputation (list, optional): List of columns to exclude from imputation
        skip_outlier_file (bool, optional): Whether to skip writing the outlier file
        skip_missing_rows_file (bool, optional): Whether to skip writing missing row file

    Returns:
        tuple: (cleaned DataFrame, cleaning statistics)
    """
    if id_columns is None:
        id_columns = []
    if exclude_from_imputation is None:
        exclude_from_imputation = []

    # Load data
    df = pd.read_csv(input_file)

    # Initialize cleaning statistics
    cleaning_stats = {
        "missing_filled": 0,
        "outliers_found": 0
    }

    # missing values handling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in id_columns and col not in exclude_from_imputation]

    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in id_columns and col not in exclude_from_imputation]

    # save rows with missing values before imputation - in case user wants to validate
    missing_rows = df[df[numeric_cols + categorical_cols].isnull().any(axis=1)]
    if not missing_rows.empty and not skip_missing_rows_file:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        missing_file = output_dir / f'missing_value_rows_{timestamp}.csv'
        missing_rows.to_csv(missing_file, index=False)

    # in numeric columns missing values handling - using KNN imputation
    if len(numeric_cols) > 0:
        try:
            # KNNImputer
            knn_imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
        except Exception as e:
            print(f"KNN imputation failed: {str(e)}")
            try:
                # IterativeImputer bakcup
                iterative_imputer = IterativeImputer(random_state=42)
                df[numeric_cols] = iterative_imputer.fit_transform(df[numeric_cols])

            except Exception as e:
                # mean imputation backup
                for col in numeric_cols:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        df[col] = df[col].fillna(df[col].mean())
                        cleaning_stats["missing_filled"] += missing_count

    # categorical columns missing values handling
    if len(categorical_cols) > 0:
        try:
            #SimpleImputer with most_frequent strategy for categorical
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            # sum of missing values that were filled
            missing_before = df[categorical_cols].isnull().sum().sum()
            missing_after = df[categorical_cols].isnull().sum().sum()
            cleaning_stats["missing_filled"] += missing_before - missing_after

        except Exception as e:
            # mode imputation backup
            for col in categorical_cols:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                    cleaning_stats["missing_filled"] += missing_count

    # if numeric data is included, check for outliers with isolation forest
    if len(numeric_cols) > 0:
        # prepare data for outlier detection
        X = df[numeric_cols].copy()

        # initialize and fit the model
        clf = IsolationForest(contamination=outlier_contamination, random_state=42)
        outliers = clf.fit_predict(X)

        # create outlier indicator column
        df["is_outlier"] = (outliers == -1).astype(int)


        # sum total amount of outliers
        cleaning_stats["outliers_found"] = (outliers == -1).sum()

        # save outliers to file
        if cleaning_stats["outliers_found"] > 0 and not skip_outlier_file:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            outlier_file = output_dir / f'outliers_{timestamp}.csv'
            outlier_data = df[outliers == -1]
            outlier_data.to_csv(outlier_file, index=False)

        # remove outliers if indicated by user
        if remove_outliers:
            df = df[outliers != -1]
            df = df.drop(columns=['is_outlier'])

    return df, cleaning_stats
