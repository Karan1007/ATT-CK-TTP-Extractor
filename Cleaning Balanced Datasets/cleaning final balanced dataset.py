import pandas as pd

# Load datasets
final_balanced_file_path = "C:/Users/karki/OneDrive/Desktop/generative AI/final_balanced_attack_data.csv"
preprocessed_file1 = "C:/Users/karki/OneDrive/Desktop/generative AI/preprocessed_attack_data.csv"
preprocessed_file2 = "C:/Users/karki/OneDrive/Desktop/generative AI/old/preprocessed_ttp_dataset.csv"

df_final = pd.read_csv(final_balanced_file_path)
df_preprocessed1 = pd.read_csv(preprocessed_file1)
df_preprocessed2 = pd.read_csv(preprocessed_file2)

# Ensure correct column name
if "Processed Data Source" in df_preprocessed1.columns:
    df_preprocessed1.rename(columns={"Processed Data Source": "ProcessedText"}, inplace=True)

# Merge back text data
if "ProcessedText" in df_preprocessed1.columns and "ProcessedText" in df_preprocessed2.columns:
    text_data = pd.concat([df_preprocessed1[["ProcessedText"]], df_preprocessed2[["ProcessedText"]]], ignore_index=True)
else:
    raise ValueError("ProcessedText column missing in preprocessed datasets")

df_final.insert(0, "ProcessedText", text_data["ProcessedText"][:len(df_final)].fillna(""))

# Remove float64 columns (unnecessary precision)
float_columns = df_final.select_dtypes(include=['float64']).columns
df_final[float_columns] = df_final[float_columns].astype(int)

# Reduce dimensionality: Remove low-variance features
threshold = 0.01  # Minimum fraction of samples that must have a nonzero value
cols_to_keep = df_final.columns[df_final.astype(bool).mean() > threshold]
df_final = df_final[cols_to_keep]

# Save the cleaned dataset
df_final.to_csv("cleaned_balanced_attack_data.csv", index=False)

print("✅ Dataset fixed! File saved as cleaned_balanced_attack_data.csv")
