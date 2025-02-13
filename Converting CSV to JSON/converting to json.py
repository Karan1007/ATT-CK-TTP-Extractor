import pandas as pd
import json

# Load the preprocessed dataset
file_path = "C:/Users/karki/OneDrive/Desktop/generative AI/cleaned_balanced_attack_data.csv"
df = pd.read_csv(file_path)

# Ensure the dataset has correct structure
df = df.dropna().reset_index(drop=True)

# Extract MITRE ATT&CK techniques dynamically
technique_columns = df.columns[1:]  # Exclude "ProcessedText"
jsonl_data = []

for _, row in df.iterrows():
    # Get techniques where value is 1
    detected_techniques = [tech for tech in technique_columns if row[tech] == 1]
    
    # If no techniques detected, say "No known technique detected"
    technique_response = ", ".join(detected_techniques) if detected_techniques else "No known technique detected"

    # Construct JSONL format
    jsonl_data.append({
        "messages": [
            {"role": "system", "content": "You are a cybersecurity AI trained to analyze threat intelligence reports."},
            {"role": "user", "content": f"Analyze this threat report: {row['ProcessedText']}"},
            {"role": "assistant", "content": f"The threat corresponds to MITRE ATT&CK techniques: {technique_response}"}
        ]
    })

# Save as JSONL for OpenAI fine-tuning
output_file = "C:/Users/karki/OneDrive/Desktop/generative AI/fine_tune_data_fixed.jsonl"
with open(output_file, "w") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Dataset converted to JSONL and saved as {output_file}")
