'''
This script converts .xlsx files to .csv files in the HondaDataSupport folder.
'''

import os
from tqdm import tqdm
import pandas as pd
import warnings

path_raw_honda = './RawData/SHRP2/HondaDataSupport/'


print('Converting .xlsx files to .csv files...')

path_xlsx = path_raw_honda + 'Insight Tables/'
path_csv = path_raw_honda + 'InsightTables_csv/'
os.makedirs(path_csv, exist_ok=True)
warnings.simplefilter(action='ignore', category=FutureWarning) # suppress FutureWarnings by this line
warnings.simplefilter(action='ignore', category=UserWarning) # suppress UserWarnings by this line

for filename in tqdm(os.listdir(path_xlsx)):
    if filename.endswith('.xlsx'):
        if filename.endswith('Dictionary.xlsx'):
            continue
        if os.path.exists(os.path.join(path_csv, filename.replace('.xlsx', '.csv'))):
            continue
        # Construct full file path
        file_path = os.path.join(path_xlsx, filename)
        # Read the first sheet of the excel file
        excel_data = pd.read_excel(file_path, sheet_name=0)
        # Convert data types
        excel_data = excel_data.convert_dtypes()
        # Generate a new file name with .csv extension
        csv_filename = filename.replace('.xlsx', '.csv')
        csv_file_path = os.path.join(path_csv, csv_filename)
        # Save the data to a CSV file
        excel_data.to_csv(csv_file_path, index=False)
        print(f'\n Converted {filename} to {csv_filename}')

