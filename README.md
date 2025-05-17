# Hotel Data Extractor & RAG Chatbot

A powerful tool to extract hotel information from websites, organize the data into structured formats, and provide an interactive Retrieval-Augmented Generation (RAG) chatbot interface for querying hotel details.

---

## 🌟 Features

- **Web Scraping:** Automatically extract hotel data from websites using Crawl4AI.
- **Structured Data:** Convert raw markdown into clean JSON containing room details, prices, and amenities.
- **Local Storage:** Store all data locally without relying on external databases.
- **RAG Chatbot:** Query hotel information naturally using a conversational chatbot.
- **Gemini Integration:** Powered by Google’s Gemini 1.5 Flash model for enhanced language understanding.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Manansharma123/Hotel_Project
cd Project_chatbot

# Create a Python virtual environment
python -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
```pip install -r requirements.txt```


🔧 Usage

1. Extract Hotel Data
```python hotel_scraper.py```
This will:

Scrape hotel websites
Extract structured hotel information
Save raw markdown files to raw_data/
Save extracted JSON files to extracted_data/
Generate a combined JSON file hotel_data_results.json
2. Create Embeddings
python create_embeddings.py
3. Run Chatbot Interface
```streamlit run hotel_search.py```
📁 Project Structure

Project_chatbot/
├── hotel_scraper.py        # Main scraper script
├── create_embeddings.py    # Script to create vector embeddings
├── hotel_search.py         # Streamlit chatbot interface
├── assets.py               # Constants and configurations
├── api_management.py       # API key management utilities
├── llm_calls.py            # Logic for LLM interactions
├── markdown.py             # Markdown processing utilities
├── utils.py                # Helper functions
├── raw_data/               # Raw markdown files storage
├── extracted_data/         # Extracted JSON data storage
├── hotel_embeddings/       # Vector embeddings storage
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not committed)
