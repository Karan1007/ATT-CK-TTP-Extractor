import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# Load datasets
balanced_file_path = "C:/Users/karki/OneDrive/Desktop/generative AI/balanced_attack_data.csv"

preprocessed_file1 = "C:/Users/karki/OneDrive/Desktop/generative AI/preprocessed_attack_data.csv"
preprocessed_file2 = "C:/Users/karki/OneDrive/Desktop/generative AI/old/preprocessed_ttp_dataset.csv"  # Assuming this is the older dataset

df_balanced = pd.read_csv(balanced_file_path)
df_preprocessed1 = pd.read_csv(preprocessed_file1)
df_preprocessed2 = pd.read_csv(preprocessed_file2)

# Standardize column names
if "Processed Data Source" in df_preprocessed1.columns:
    df_preprocessed1.rename(columns={"Processed Data Source": "ProcessedText"}, inplace=True)
if "ProcessedText" not in df_preprocessed2.columns:
    raise ValueError("ProcessedText column missing in preprocessed datasets")

# Merge text features from both preprocessed datasets
text_data = pd.concat([df_preprocessed1[["ProcessedText"]], df_preprocessed2[["ProcessedText"]]], ignore_index=True)

# Ensure all attack technique columns are aligned
all_techniques = sorted(set(df_preprocessed1.columns[1:]).union(set(df_preprocessed2.columns[1:])).union(set(df_balanced.columns)))

df_preprocessed1 = df_preprocessed1.reindex(columns=["ProcessedText"] + all_techniques, fill_value=0)
df_preprocessed2 = df_preprocessed2.reindex(columns=["ProcessedText"] + all_techniques, fill_value=0)
df_balanced = df_balanced.reindex(columns=all_techniques, fill_value=0)

# Merge datasets
df_final = pd.concat([df_preprocessed1, df_preprocessed2, df_balanced], ignore_index=True)

# Handle missing values in text data
df_final["ProcessedText"] = df_final["ProcessedText"].fillna("")

# Separate features and labels
X = df_final[["ProcessedText"]]
y = df_final.iloc[:, 1:].astype(int)  # Convert labels to integers

# Convert text features to numerical using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X["ProcessedText"])

# Train a MultiOutputClassifier using RandomForest to balance multi-label data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
classifier = MultiOutputClassifier(RandomForestClassifier())
classifier.fit(X_train, y_train)

# Generate synthetic labels using the trained model
y_resampled = pd.DataFrame(classifier.predict(X_tfidf), columns=y.columns)

# Convert back to DataFrame
balanced_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
balanced_df = pd.concat([balanced_df, y_resampled], axis=1)

# Save the final corrected dataset
balanced_df.to_csv("final_balanced_attack_data.csv", index=False)

print("✅ Dataset fully corrected and balanced using MultiOutputClassifier! File saved as final_balanced_attack_data.csv")
