Command Line Arguments

Required:
input_file: the path to input csv file. Must be a .csv file.

Optional:
--output-dir: specify a folder to save output files. The default folder is named 'output'.
--id-columns: one or more column names that will be treated as unique. Separated with a space. No comma required.
--exclude-from-imputation: one or more column names that will not have values imputed during the cleaning process.
--outlier-contamination: expected proportion of outliers in the data set. A float value between (0.0 to 0.5]. The default value = 0.05
--remove-outliers: if this flag is included, outliers will be removed from the cleaned data set.
--target-column: this flag is for the column that is the target value

If the user just wants the analysis displayed to the console, and nothing saved to the computer:
--skip-analysis-file: does not save analytical report .txt file
--skip-cleaned-data-file: does not save cleaned data .csv file
--skip-missing-rows-file: does not save the missing value rows .csv file
--skip-outlier-file: does not save outliers .csv file
--skip-summary-overview-file: does not save summary overview .txt file

Examples: 
python3 src/main.py data/iris.csv --target-column species --remove-outliers --outlier-contamination 0.01
python3 src/main.py data/titanic.csv --id-columns PassengerId Name Ticket --remove-outliers --outlier-contamination 0.05 --exclude-from-imputation Cabin
