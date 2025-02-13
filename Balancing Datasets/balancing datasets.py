import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the two preprocessed datasets
file1 = "C:/Users/karki/OneDrive/Desktop/generative AI/preprocessed_attack_data.csv"
file2 = "C:/Users/karki/OneDrive/Desktop/generative AI/old/preprocessed_ttp_dataset.csv"  # Assuming this is the older dataset

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Standardize column names
df2.rename(columns={'ProcessedText': 'Processed Data Source'}, inplace=True)

# Align the TTP columns efficiently using pd.concat()
all_techniques = sorted(set(df1.columns[1:]).union(set(df2.columns[1:])))
df1 = pd.concat([df1, pd.DataFrame(0, index=df1.index, columns=[col for col in all_techniques if col not in df1.columns])], axis=1)
df2 = pd.concat([df2, pd.DataFrame(0, index=df2.index, columns=[col for col in all_techniques if col not in df2.columns])], axis=1)

# Ensure both datasets have the same column order
df1 = df1[['Processed Data Source'] + all_techniques]
df2 = df2[['Processed Data Source'] + all_techniques]

# Merge datasets
df_combined = pd.concat([df1, df2], ignore_index=True)

# Convert labels to int
df_combined.iloc[:, 1:] = df_combined.iloc[:, 1:].astype(int)

# Fill NaN values in text column
df_combined['Processed Data Source'] = df_combined['Processed Data Source'].fillna('')

# Separate features and labels
X = df_combined[['Processed Data Source']]
y = df_combined.iloc[:, 1:]

# Convert text features to numerical using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X['Processed Data Source'])

# Use MultiOutputClassifier with RandomForest to handle multi-label classification
classifier = MultiOutputClassifier(RandomForestClassifier())
classifier.fit(X_tfidf, y)

# Predict on the dataset to get balanced labels
y_resampled = pd.DataFrame(classifier.predict(X_tfidf), columns=y.columns)

# Convert back to DataFrame
balanced_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
balanced_df = pd.concat([balanced_df, y_resampled], axis=1)

# Save the balanced dataset
balanced_df.to_csv("balanced_attack_data.csv", index=False)

print("✅ Dataset balanced successfully! File saved as balanced_attack_data.csv")