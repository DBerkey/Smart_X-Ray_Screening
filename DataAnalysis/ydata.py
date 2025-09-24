import pandas as pd
from ydata_profiling import ProfileReport


source_path = 'path/to/Data_Entry_2017_v2020.csv'  # Replace with the path to your CSV file
store_path = 'path/to/store/report'  # Replace with the path to store the report


df = pd.read_csv(source_path)
new_df = pd.DataFrame()

for index, row in df.iterrows():
    findings = row['Finding Labels'].strip().split('|')
    for finding in findings:
        new_row = row.copy()
        new_row['Finding Labels'] = finding
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

profile = ProfileReport(new_df, title="Personal Info Data Profiling Report")
profile.to_file(store_path + "/personal_info_data_profiling_report.html")