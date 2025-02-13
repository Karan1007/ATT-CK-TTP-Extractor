# ATT&CK TTP Extractor

## 1. Project Overview

### What is this project?

\# ATT-CK-TTP-Extractor

This project is an ATT&amp;CK TTP Extractor, designed to scrape, preprocess, balance, and fine-tune threat intelligence data using MITRE ATT&amp;CK techniques. It also provides a **Flask API** and a **web interface** for extracting and analyzing threat techniques from security reports and uploaded documents.

### Why have I used these components?

- **Data Scraping**: MITRE ATT&CK data is scraped from online sources to create a dataset of attack techniques and threat intelligence. This allows for up-to-date information on cyber threats.
- **Data Preprocessing**: Cleaning and transforming the dataset to extract meaningful features. This step ensures that unnecessary characters, stopwords, and irrelevant data are removed for better model training.
- **Balancing Data**: Using **TF-IDF and Random Forest-based MultiOutputClassifier** to balance the dataset for better model training. Some techniques occur less frequently in datasets, so oversampling helps mitigate imbalance issues.
- **Converting CSV to JSON**: The dataset is converted into **JSONL (JSON Lines) format** to be used in fine-tuning the LLM model. JSONL is required for OpenAI-based fine-tuning as it structures training data into message-based conversation formats.
- **Fine-tuning LLM**: Using **GPT-Neo 125M**, a small and efficient transformer-based model, to fine-tune the dataset on ATT&CK techniques. GPT-Neo was chosen because it provides a balance between computational efficiency and effectiveness in generating structured responses.
- **Flask API**: Provides an endpoint to analyze uploaded security reports. Flask was chosen due to its lightweight nature and ease of integrating machine learning models.
- **Web Interface**: Simple front-end for user interaction, allowing easy file uploads and automated threat analysis.

### Project Structure

The project is organized into 10 folders:

1. **Dataset**: Contains the original dataset.
2. **Preprocessed Dataset**: Python scripts and CSV files after cleaning raw data.
3. **Scraping Data**: Scripts for scraping MITRE ATT&CK techniques.
4. **Preprocessing Scraped Dataset**: Scripts for cleaning scraped data.
5. **Balancing Datasets**: Balances datasets using machine learning models.
6. **Cleaning Balanced Datasets**: Removes unnecessary data and ensures dataset consistency.
7. **Converting CSV to JSON**: Converts data to JSON format for fine-tuning. JSON format is required as fine-tuning language models use structured conversation-like training data.
8. **Fine-tuning**: Fine-tunes a language model for better ATT&CK technique predictions. The fine-tuning process helps the model better understand cybersecurity reports and recommend relevant MITRE techniques.
9. **Flask**: The backend API that takes user input and analyzes threats.
10. **Interface**: The frontend for uploading and analyzing reports.

---

## 2. How to Run

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Flask
- PyTorch
- Transformers (Hugging Face)
- pandas, scikit-learn, nltk
- BeautifulSoup (for web scraping)

### Installation Steps

1. Clone the repository:

   ```sh
   git clone https://github.com/Karan1007/attack-ttp-extractor.git
   cd attack-ttp-extractor
   ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Run the **Flask API**:

   ```sh
   cd flask
   python app.py
   ```

4. Open the **Web Interface**:

   - Open `interface/index.html` in a browser.
   - Upload a security report (PDF/TXT) and analyze it.

5. Fine-tune the model (optional, if retraining is required):

   ```sh
   cd fine-tuning
   python training_with_llm.py
   ```

### Expected Output

- A JSON response with threat analysis when using the Flask API.
- A summarized threat report with extracted ATT&CK techniques on the web interface.

---

### Future Improvements

- Enhancing dataset diversity
- Adding more scraping sources
- Improving model accuracy with fine-tuning on additional security reports

### Why GPT-Neo 125M?

GPT-Neo 125M was selected because it provides a **lighter, computationally efficient alternative** to large-scale models like GPT-3, while still being powerful enough for our cybersecurity analysis needs. Fine-tuning allows the model to better understand **MITRE ATT&CK techniques** and generate useful cybersecurity intelligence.

### Why JSON format for fine-tuning?

Fine-tuning models, particularly those trained for **NLP-based security intelligence**, require structured input. JSONL format is ideal because it allows the model to process **system prompts, user queries, and assistant responses in a structured conversational format**. This enhances the model's ability to map user-provided security reports to MITRE techniques efficiently.

### Why preprocess and balance datasets?

Security datasets often suffer from **class imbalance**, where some attack techniques occur far more frequently than others. To ensure the model learns effectively, balancing techniques such as **TF-IDF vectorization and synthetic oversampling** are applied to distribute the attack techniques evenly across the dataset.

