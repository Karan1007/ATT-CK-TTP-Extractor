import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Load the existing dataset
df = pd.read_csv("C:/Users/karki/OneDrive/Desktop/generative AI/mitre_attack_datasources.csv")

def get_attack_techniques(data_source_url):
    """Scrapes ATT&CK techniques linked to a data source."""
    response = requests.get(data_source_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find techniques (stored in tables on MITRE pages)
    table = soup.find('table', class_='table')
    techniques = []
    if table:
        for row in table.find_all('tr')[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 2:
                technique_id = cols[0].text.strip()
                technique_name = cols[1].text.strip()
                techniques.append(f"{technique_id} - {technique_name}")
    
    return ", ".join(techniques) if techniques else "None"

# Add a new column for ATT&CK techniques
df["ATT&CK Techniques"] = df["Link"].apply(get_attack_techniques)
    
# Save the enhanced dataset
file_path = "C:/Users/karki/OneDrive/Desktop/generative AI/enhanced_mitre_attack_datasources.csv"
df.to_csv(file_path, index=False)

print("✅ Data scraping complete! File saved as enhanced_mitre_attack_datasources.csv")
