import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
file_path = r"C:\Users\karki\OneDrive\Desktop\generative AI\CISA-crawl-rt-ttp-ct.csv"
df = pd.read_csv(file_path)

# Remove unnecessary column
df.drop(columns=["Unnamed: 0"], inplace=True)

# Function to clean text
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)  # Remove newlines
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply cleaning to CleanText column
df['ProcessedText'] = df['CleanText'].apply(clean_text)

# Convert TTP column (set of techniques) into a list
df['TTP'] = df['TTP'].apply(lambda x: re.findall(r'T\d{4}(?:\.\d{3})?', x))

# One-hot encode TTP labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
ttp_encoded = mlb.fit_transform(df['TTP'])
ttp_labels = pd.DataFrame(ttp_encoded, columns=mlb.classes_)

# Merge processed text with encoded labels
final_df = pd.concat([df[['ProcessedText']], ttp_labels], axis=1)

# Save the preprocessed dataset
final_df.to_csv("preprocessed_ttp_dataset.csv", index=False)

print("✅ Preprocessing complete! File saved as preprocessed_ttp_dataset.csv")
