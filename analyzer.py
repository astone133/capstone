import numpy as np
import pandas as pd

def analyze_data(df, id_columns=None, target_column=None):
    """
    Analyze the dataset and return statistical insights.

    Args:
        df (pd.DataFrame): Input DataFrame
        id_columns (list, optional): List of columns to treat as identifiers
        target_column (str, optional): Name of the target column for analysis

    Returns:
        dict: Dictionary containing analysis results
            'basic_stats': basic_stats,
            'distribution_stats': distribution_stats,
            'correlations': correlations,
            'high_variance_columns': high_variance_columns,
            'categorical_counts': categorical_counts,
            'high_corr_pairs': high_corr_pairs
    """
    if id_columns is None:
        id_columns = []

    # Get numeric and categorical columns (excluding identifiers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in id_columns and not col.endswith('_is_outlier')]

    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in id_columns]

    # Basic statistics for numeric columns
    basic_stats = df[numeric_cols].describe().to_dict()

    # Calculate skewness and kurtosis for numeric columns
    distribution_stats = {
        'skewness': df[numeric_cols].skew().to_dict(),
        'kurtosis': df[numeric_cols].kurtosis().to_dict()
    }

    # Calculate correlation matrix for numeric columns
    correlations = {}
    corr_matrix = df[numeric_cols].corr()
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.6:  # Only include strong correlations
                correlations[f"{numeric_cols[i]} vs {numeric_cols[j]}"] = corr

    # Multicollinearity analysis
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr_pairs = [
        (col1, col2, corr_matrix.loc[col1, col2])
        for i, col1 in enumerate(corr_matrix.columns)
        for j, col2 in enumerate(corr_matrix.columns)
        if i < j and corr_matrix.loc[col1, col2] > 0.90
    ]

    # Identify high variance columns
    high_variance_columns = []
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > mean * 0.5:  # High variance if std dev is more than 50% of mean
            high_variance_columns.append(col)

    # Value counts for categorical columns
    categorical_counts = {}
    for col in categorical_cols:
        categorical_counts[col] = df[col].value_counts().to_dict()

    return {
        'basic_stats': basic_stats,
        'distribution_stats': distribution_stats,
        'correlations': correlations,
        'high_variance_columns': high_variance_columns,
        'categorical_counts': categorical_counts,
        'high_corr_pairs': high_corr_pairs
    }
