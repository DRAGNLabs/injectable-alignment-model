import pandas as pd

def convert_xlsx_to_csv(input_xlsx_file, output_csv_file):
    # Read the Excel file
    data = pd.read_excel(input_xlsx_file)
    
    # Write the data to a CSV file
    data.to_csv(output_csv_file, index=False)

# Replace 'input_file.xlsx' and 'output_file.csv' with your file paths
input_file_path = '4_year_old_words.xlsx'
output_file_path = '4yo_words.csv'

convert_xlsx_to_csv(input_file_path, output_file_path)


