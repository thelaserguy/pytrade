import pandas as pd

# Load the CSV file
df = pd.read_csv("merged.csv")

# First forward fill to carry forward the previous day's value
df = df.fillna(method='ffill')

# Then backfill to carry backward the next day's value
df = df.fillna(method='bfill')

# Save the cleaned data
df.to_csv("cleaned.csv", index=False)
