import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the enhanced dataset
file_path = "C:/Users/karki/OneDrive/Desktop/generative AI/enhanced_mitre_attack_datasources.csv"
df = pd.read_csv(file_path)

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply cleaning to Data Source column
df['Processed Data Source'] = df['Data Source'].apply(clean_text)

# Process ATT&CK Techniques column
df['ATT&CK Techniques'] = df['ATT&CK Techniques'].fillna('')
df['Techniques List'] = df['ATT&CK Techniques'].apply(lambda x: re.findall(r'T\d{4}(?:\.\d{3})?', x))

# One-hot encode ATT&CK Techniques
mlb = MultiLabelBinarizer()
techniques_encoded = mlb.fit_transform(df['Techniques List'])
techniques_labels = pd.DataFrame(techniques_encoded, columns=mlb.classes_)

# Merge processed text with encoded techniques
final_df = pd.concat([df[['Processed Data Source']], techniques_labels], axis=1)

# Save the preprocessed dataset
final_df.to_csv("C:/Users/karki/OneDrive/Desktop/generative AI/preprocessed_attack_data.csv", index=False)

print("✅ Preprocessing complete! File saved as preprocessed_attack_data.csv")
