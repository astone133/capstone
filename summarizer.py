import numpy as np
import pandas as pd

def generate_technical_output(df, analysis_results, cleaning_stats=None, id_columns=None, target_column=None):
    """
    Generate a structured technical output of the data analysis results.

    Args:
        df (pd.DataFrame): Input DataFrame
        analysis_results (dict): Results from the analyzer module
        cleaning_stats (dict): Statistics from the cleaning process
        id_columns (list): List of columns to treat as identifiers
        target_column (str): Name of the target column for analysis

    Returns:
        str: Structured technical output
    """
    if id_columns is None:
        id_columns = []

    output_parts = []

    # Dataset Overview
    output_parts.append("DATASET OVERVIEW")
    output_parts.append("-" * 50)
    output_parts.append(f"• Total rows: {len(df)}")
    output_parts.append(f"• Total columns: {len(df.columns)}")
    if id_columns:
        output_parts.append(f"• Identifier columns: {', '.join(id_columns)}")

    # Target Class Distribution (if target column is categorical or discrete numeric)
    if target_column and target_column in df.columns:
        # Check if column is not continuous (numeric) or is discrete numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
        is_discrete = is_numeric and df[
            target_column].nunique() <= 20  # Consider numeric columns with 20 or fewer unique values as discrete

        if not is_numeric or is_discrete:
            output_parts.append("\nTARGET CLASS DISTRIBUTION")
            output_parts.append("-" * 50)
            output_parts.append("frequency    percent")
            class_counts = df[target_column].value_counts().sort_values(ascending=False)
            total = len(df)
            # Get top 10 classes
            top_classes = class_counts.head(10)
            for class_name, count in top_classes.items():
                percentage = (count / total) * 100
                output_parts.append(f"• {class_name}: {count} ({percentage:.1f}%)")
            # Add note if there are more classes
            if len(class_counts) > 10:
                remaining_count = class_counts[10:].sum()
                remaining_percentage = (remaining_count / total) * 100
                output_parts.append(f"• Other: {remaining_count} ({remaining_percentage:.1f}%)")

    # Cleaning Summary
    if cleaning_stats:
        output_parts.append("\nDATA CLEANING")
        output_parts.append("-" * 50)
        if cleaning_stats["missing_filled"] > 0:
            output_parts.append(f"• Missing values filled: {cleaning_stats['missing_filled']}")
        if cleaning_stats["outliers_found"] > 0:
            output_parts.append(f"• Outliers detected: {cleaning_stats['outliers_found']}")

    # Categorical Columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in id_columns]
    if len(categorical_cols) > 0:
        output_parts.append("\nCATEGORICAL COLUMNS")
        output_parts.append("-" * 50)
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            most_common = df[col].mode().iloc[0]
            output_parts.append(f"• {col}:")
            output_parts.append(f"  - Unique values: {unique_vals}")
            output_parts.append(f"  - Most common value: '{most_common}'")

    # Numeric Columns Statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in id_columns]
    if len(numeric_cols) > 0:
        output_parts.append("\nNUMERIC COLUMNS")
        output_parts.append("-" * 50)
        for col in numeric_cols:
            stats = analysis_results['basic_stats'][col]
            output_parts.append(f"• {col}:")
            output_parts.append(f"  - Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
            output_parts.append(f"  - Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
            output_parts.append(f"  - Std Dev: {stats['std']:.2f}")

    # Distribution Statistics
    if 'distribution_stats' in analysis_results:
        output_parts.append("\nDISTRIBUTION ANALYSIS")
        output_parts.append("-" * 50)
        dist_info = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in (id_columns or []):
                skewness = analysis_results['distribution_stats']['skewness'].get(col, 0)
                kurtosis = analysis_results['distribution_stats']['kurtosis'].get(col, 0)
                if abs(skewness) > 1 or abs(kurtosis) > 3:
                    dist_info.append(
                        f"The {col}: distribution shows significant skewness ({skewness:.2f}) and kurtosis of {kurtosis:.2f}")
        if dist_info:
            output_parts.append("\n".join(dist_info) + ".")

    # Correlations
    if 'correlations' in analysis_results and analysis_results['correlations']:
        output_parts.append("\nSTRONG CORRELATIONS")
        output_parts.append("-" * 50)
        for pair, value in analysis_results['correlations'].items():
            output_parts.append(f"• {pair}: {value}")

    # Multicollinearity Analysis
    if 'high_corr_pairs' in analysis_results and analysis_results['high_corr_pairs']:
        output_parts.append("\nMULTICOLLINEARITY ANALYSIS")
        output_parts.append("-" * 50)
        output_parts.append("Highly correlated feature pairs (|correlation| > 0.90):")
        for col1, col2, corr in analysis_results['high_corr_pairs']:
            output_parts.append(f"• {col1} vs {col2}: {corr:.3f}")
    elif 'high_corr_pairs' in analysis_results:
        output_parts.append("\nMULTICOLLINEARITY ANALYSIS")
        output_parts.append("-" * 50)
        output_parts.append("No significant multicollinearity detected (|correlation| > 0.90)")

    # High Variance Columns
    if 'high_variance_columns' in analysis_results:
        high_var_cols = analysis_results['high_variance_columns']
        if high_var_cols:
            output_parts.append("\nHIGH VARIANCE COLUMNS")
            output_parts.append("-" * 50)
            output_parts.append(f"• {', '.join(high_var_cols)}")

    return "\n".join(output_parts)


def generate_template_summary(technical_output: str) -> str:
    """
    Generate a template-based summary of the technical output.

    Args:
        technical_output (str): The structured technical output

    Returns:
        str: Template-based summary
    """
    sections = technical_output.split("\n\n")
    summary_parts = []

    # dataset overview
    if "DATASET OVERVIEW" in sections[0]:
        overview = sections[0].split("DATASET OVERVIEW")[1].strip()
        lines = overview.split("\n")
        rows = cols = None
        for line in lines:
            if "Total rows:" in line:
                rows = line.split(":")[1].strip()
            elif "Total columns:" in line:
                cols = line.split(":")[1].strip()
        if rows and cols:
            summary_parts.append(f"The dataset contains {rows} rows and {cols} columns.")

    # cleaning information
    if "DATA CLEANING" in technical_output:
        cleaning_section = [s for s in sections if "DATA CLEANING" in s][0]
        lines = cleaning_section.split("\n")
        outliers = None
        for line in lines:
            if "Outliers detected:" in line:
                outliers = line.split(":")[1].strip()
        if outliers:
            summary_parts.append(f"During data cleaning, {outliers} outliers were detected.")

    # distribution analysis
    if "DISTRIBUTION ANALYSIS" in technical_output:
        dist_section = [s for s in sections if "DISTRIBUTION ANALYSIS" in s][0]
        dist_info = []
        for line in dist_section.split("\n"):
            if "distribution shows significant skewness" in line:
                dist_info.append(line.strip().rstrip('.'))
        if dist_info:
            summary_parts.append(" ".join(dist_info) + ".")

    # correlations
    if "STRONG CORRELATIONS" in technical_output:
        corr_section = [s for s in sections if "STRONG CORRELATIONS" in s][0]
        pairs = []
        for line in corr_section.split("\n"):
            if line.startswith("•"):
                pair, value = line[2:].split(":")
                pair = pair.replace("vs", "and").strip()
                value = float(value.strip())
                pairs.append(f"{pair} ({value:.3f})")
        if pairs:
            joined = ", ".join(pairs[:-1]) + f", and {pairs[-1]}" if len(pairs) > 2 else " and ".join(pairs)
            summary_parts.append(f"Strong correlations were found between {joined}.")

    # multicollinearity
    if "MULTICOLLINEARITY ANALYSIS" in technical_output:
        multi_section = [s for s in sections if "MULTICOLLINEARITY ANALYSIS" in s][0]
        multi_pairs = []
        for line in multi_section.split("\n"):
            if line.startswith("•"):
                try:
                    pair, value = line[2:].split(":")
                    pair = pair.replace("vs", "and").strip()
                    value = float(value.strip())
                    multi_pairs.append(f"{pair} ({value:.3f})")
                except ValueError:
                    continue
        if multi_pairs:
            if len(multi_pairs) > 2:
                joined = ", ".join(multi_pairs[:-1]) + f", and {multi_pairs[-1]}"
            else:
                joined = " and ".join(multi_pairs)
            summary_parts.append(f"Multicollinearity was detected between {joined}.")

    # high variance columns
    if "HIGH VARIANCE COLUMNS" in technical_output:
        var_section = [s for s in sections if "HIGH VARIANCE COLUMNS" in s][0]
        var_info = []
        for line in var_section.split("\n"):
            if line.startswith("•"):
                cols = line[2:].strip()
                var_info.append(f"Columns that show high variance: {cols}")
        if var_info:
            summary_parts.append(" ".join(var_info) + ".")

    return "\n".join(summary_parts)

def generate_summary(df, analysis_results, cleaning_stats=None, id_columns=None, target_column=None):
    """
    Generate both technical output and NLP summary of the data analysis results.

    Args:
        df (pd.DataFrame): Input DataFrame
        analysis_results (dict): Results from the analyzer module
        cleaning_stats (dict): Statistics from the cleaning process
        id_columns (list): List of columns to treat as identifiers
        target_column (str): Name of the target column for analysis

    Returns:
        tuple: (technical_output, nlp_summary)
    """
    # Generate technical output
    technical_output = generate_technical_output(df, analysis_results, cleaning_stats, id_columns, target_column)

    # Generate NLP summary using T5


    return technical_output

def generate_structured_summary(df, analysis_results, cleaning_stats=None, id_columns=None) -> str:
    """
    Generate a concise, structured summary of the dataset analysis.

    Args:
        df (pd.DataFrame): Cleaned DataFrame
        analysis_results (dict): Results from the analyzer module
        cleaning_stats (dict): Statistics from the cleaning process
        id_columns (list): List of columns to treat as identifiers

    Returns:
        str: A short, human‑readable summary
    """
    summary_parts = []

    # Basic dataset info
    rows, cols = len(df), len(df.columns)
    summary_parts.append(f"The dataset contains {rows} rows and {cols} columns.")

    # Cleaning information
    if cleaning_stats:
        ot = cleaning_stats.get("outliers_found", 0)
        if ot > 0:
            summary_parts.append(f"During data cleaning, {ot} outliers were detected.")
        summary_parts.append("")  # Add empty line after cleaning info

    # Distribution analysis
    if 'distribution_stats' in analysis_results:
        dist_info = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in (id_columns or []):
                skewness = analysis_results['distribution_stats']['skewness'].get(col, 0)
                kurtosis = analysis_results['distribution_stats']['kurtosis'].get(col, 0)
                if abs(skewness) > 1 or abs(kurtosis) > 3:
                    dist_info.append(
                        f"The {col}: distribution shows significant skewness ({skewness:.2f}) and kurtosis of {kurtosis:.2f}")
        if dist_info:
            summary_parts.append(" ".join(dist_info) + ".")

    # High‑variance columns
    hv = analysis_results.get("high_variance_columns", [])
    if hv:
        hv_list = ", ".join(hv)
        summary_parts.append(f"{hv_list} show high variance in their values.")

    return "\n".join(summary_parts)
