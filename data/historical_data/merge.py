import pandas as pd
import os

# Set the path to the current directory
path = '.'

# List all CSV files in the directory
files = [file for file in os.listdir(path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dfs = []

# Iterate over each file
for file in files:
    print(f'Processing {file}...')
    try:
        # Load the CSV file
        data = pd.read_csv(os.path.join(path, file))
    except Exception as e:
        print(f"Could not read file {file} due to {e}")
        continue

    # Check for necessary columns
    if 'Date' not in data.columns or 'Close' not in data.columns:
        print(f"File {file} does not have 'Date' and 'Close' columns")
        continue

    try:
        # Convert the 'Date' column to datetime type
        data['Date'] = pd.to_datetime(data['Date'])
    except Exception as e:
        print(f"Could not parse dates in file {file} due to {e}")
        continue

    # Set the 'Date' column as the index
    data.set_index('Date', inplace=True)

    # Add the 'Close' column DataFrame to the list
    # We use the file name (without .csv) as the column name
    dfs.append(data[['Close']].rename(columns={'Close': file.replace('.csv', '')}))

try:
    # Concatenate all dataframes along the columns
    df = pd.concat(dfs, axis=1)

    # Save the merged data to a new CSV file
    df.to_csv('merged.csv')
except Exception as e:
    print(f"Could not save merged data due to {e}")
