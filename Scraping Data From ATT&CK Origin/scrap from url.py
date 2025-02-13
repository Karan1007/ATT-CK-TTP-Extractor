import requests
from bs4 import BeautifulSoup
import pandas as pd

# MITRE ATT&CK Data Sources URL
URL = "https://attack.mitre.org/datasources/"

# Send request & parse HTML
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all data source entries
data_sources = soup.find_all('tr')[1:]  # Skip header row

# Extract data
scraped_data = []
for row in data_sources:
    cols = row.find_all('td')
    if len(cols) >= 2:
        name = cols[0].text.strip()
        description = cols[1].text.strip()
        link = "https://attack.mitre.org" + cols[0].find('a')['href'] if cols[0].find('a') else "N/A"
        scraped_data.append([name, description, link])

# Convert to DataFrame
df = pd.DataFrame(scraped_data, columns=["Data Source", "Description", "Link"])

# Save to CSV
csv_filename = "mitre_attack_datasources.csv"
df.to_csv(csv_filename, index=False)

print(f"✅ Data scraped successfully! File saved as {csv_filename}")
