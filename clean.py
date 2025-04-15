import pandas as pd

# Load the CSV file with low_memory set to False
df = pd.read_csv('Phishing_email.csv', low_memory=False)

# Remove rows where either 'body' or 'label' is blank (NaN or empty string)
df_cleaned = df.dropna(subset=['body', 'label'])  # Remove rows with NaN values
df_cleaned = df_cleaned[(df_cleaned['body'] != '') & (df_cleaned['label'] != '')]  # Remove rows with empty strings

# Filter out non-numeric 'label' values
df_cleaned = df_cleaned[pd.to_numeric(df_cleaned['label'], errors='coerce').notnull()]

# Convert 'label' column to integers
df_cleaned['label'] = df_cleaned['label'].astype(int)

# Save the cleaned DataFrame back to CSV
df_cleaned.to_csv('Phishing_email.csv', index=False)

print("Rows with blank 'body' or 'label' have been removed.")