#!/usr/bin/env python3
import numpy as np
import pandas as pd
from cleaner import clean_data
from analyzer import analyze_data
from summarizer import generate_summary, generate_template_summary, generate_structured_summary
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Data Analysis and Cleaning Tool')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--id-columns', nargs='+', help='List of columns to treat as identifiers')
    parser.add_argument('--exclude-from-imputation', nargs='+', help='List of columns to exclude from imputation')
    parser.add_argument('--outlier-contamination', type=float, default=0.05,
                        help='Expected proportion of outliers in the dataset (0-0.5)')
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Remove outliers from the dataset')
    parser.add_argument('--target-column', help='Name of the target column for analysis')
    parser.add_argument('--skip-analysis-file', action='store_true', help='Skip writing the analysis report .txt file')
    parser.add_argument('--skip-outlier-file', action='store_true', help='Skip writing the outliers .csv file')
    parser.add_argument('--skip-cleaned-data-file', action='store_true', help='Skip writing the cleaned data .csv file')
    parser.add_argument('--skip-summary-overview-file', action='store_true', help='Skip writing the summary overview .txt file')
    parser.add_argument('--skip-missing-rows-file', action='store_true', help='Skip writing the missing value rows .csv file')
    args = parser.parse_args()

    input_file = args.input_file

    # validate if file exists
    while not os.path.isfile(input_file):
        new_file_name =input(
        f"Error: '{input_file}' does not exist.\n"
        "Type a CSV file name or type 'exit' to quit: ").strip()
        if new_file_name.lower() in ('exit', 'quit'):
            print("Program exited.")
            sys.exit(1)
        input_file = new_file_name

    args.input_file = input_file


    # validate dataset is a comma separated value file
    # if not csv, type name again or exit program
    while not input_file.lower().endswith('.csv'):
        new_file_name =input(
        "Error: File must have a .csv extension.\n"
        "Please retype the CSV file name or type 'exit' to quit: ").strip()
        if new_file_name.lower() in ('exit', 'quit'):
            print("Program exited.")
            sys.exit(1)
        input_file = new_file_name

    args.input_file = input_file

    # if csv file is size 0, enter new file name, or exit program
    while os.path.getsize(input_file) == 0:
        new_file_name = input(
            f"Error: '{input_file}' is empty.\n"
            "Please enter a different CSV file name or type 'exit' to quit: "
        ).strip()
        if new_file_name.lower() in ('exit', 'quit'):
            print("Program exited.")
            sys.exit(1)
        input_file = new_file_name

    args.input_file = input_file

    # create output directory if doesn't already exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # initial loading information
        print(f"Loading data from {args.input_file}")

        # shape information
        initial_df = pd.read_csv(args.input_file)
        print(f"Loaded DataFrame shape: {initial_df.shape}")

        # identify column types
        numeric_cols = initial_df.select_dtypes(include=[np.number]).columns
        categorical_cols = initial_df.select_dtypes(include=['object']).columns
        print(f"Numeric columns: {list(numeric_cols)}")
        print(f"Categorical columns: {list(categorical_cols)}")

        # check target column exists if specified
        if args.target_column and args.target_column not in initial_df.columns:
            raise ValueError(f"Target column '{args.target_column}' not found in dataset")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # load and clean data and create csv files
        df, cleaning_stats = clean_data(
            timestamp,
            args.input_file,
            args.id_columns,
            args.outlier_contamination,
            args.remove_outliers,
            args.exclude_from_imputation,
            args.skip_outlier_file,
            args.skip_missing_rows_file
        )

        # print cleaning results
        print(f"Missing values after cleaning: {cleaning_stats['missing_filled']}")
        print(f"Detecting outliers using Isolation Forest (contamination={args.outlier_contamination * 100}%)")
        print(f"Outliers detected: {cleaning_stats['outliers_found']}")
        if args.remove_outliers:
            print(f"Rows removed: {cleaning_stats['outliers_found']}")
        print(f"Returning DataFrame shape: {df.shape}")

        # create analysis DataFrame excluding is_outlier column if it exists
        if 'is_outlier' in df.columns:
            analysis_df = df.drop(columns=['is_outlier']).copy()
        else:
            analysis_df = df.copy()

        # analyze data from analyzer.py
        analysis_results = analyze_data(analysis_df, args.id_columns, args.target_column)

        # generate summaries from summarizer.py
        technical_output = generate_summary(analysis_df, analysis_results, cleaning_stats, args.id_columns,
                                                         args.target_column)
        template_summary = generate_template_summary(technical_output)
        structured_summary = generate_structured_summary(analysis_df, analysis_results, cleaning_stats, args.id_columns)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save cleaned data
        if not args.skip_cleaned_data_file:
            cleaned_file = output_dir / f'cleaned_data_{timestamp}.csv'
            df.to_csv(cleaned_file, index=False)

        # Save technical output txt
        technical_output_file = output_dir / f'analysis_report_{timestamp}.txt'
        with open(technical_output_file, 'w') as f:
            f.write(technical_output)

        # Save template summary txt
        template_summary_file = output_dir / f'summary_overview_{timestamp}.txt'
        with open(template_summary_file, 'w') as f:
            f.write(template_summary)

        print("\nAnalysis complete! Results saved in output/")
        if not args.skip_cleaned_data_file:
            print(f"• Cleaned data: cleaned_data_{timestamp}.csv")
        if not args.skip_analysis_file:
            print(f"• Technical output: analysis_report_{timestamp}.txt")
        if not args.skip_summary_overview_file:
            print(f"• Summary Overview: summary_overview_{timestamp}.txt")
        if not args.skip_outlier_file:
            outliers_file = output_dir / f'outliers_{timestamp}.csv'
            if outliers_file.exists():
                print(f"• Outliers: {outliers_file.name}")
        if not args.skip_missing_rows_file:
            missing_file = output_dir / f'missing_value_rows_{timestamp}.csv'
            if missing_file.exists():
                print(f"• Missing value rows: {missing_file.name}")

        print("\nTechnical Output:")
        print("=" * 50)
        print(technical_output)

        print("\nStructured Summary:")
        print("=" * 50)
        print(structured_summary)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
