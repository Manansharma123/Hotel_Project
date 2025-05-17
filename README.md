Hotel Data Extractor & RAG Chatbot
A tool that extracts hotel information from websites, saves structured data, and provides a RAG-powered chatbot interface for querying hotel details.

🌟 Features
Web Scraping: Extract hotel data from websites using crawl4ai

Structured Data: Process raw markdown into clean JSON with room details, prices, amenities

Local Storage: Store all data locally without external databases

RAG Chatbot: Query hotel information using natural language

Gemini Integration: Powered by Google's Gemini 1.5 Flash model

🚀 Quick Start
Installation
bash
# Clone repository
git clone https://github.com/yourusername/hotel-data-extractor.git
cd hotel-data-extractor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Configuration
Create a .env file with your API key:

text
GOOGLE_API_KEY=your_gemini_api_key_here
Usage
1. Extract Hotel Data
bash
python hotel_scraper.py
This will:

Scrape hotel websites

Extract structured information

Save raw markdown in raw_data/

Save JSON data in extracted_data/

Create combined hotel_data_results.json

2. Create Embeddings
bash
python create_embeddings.py
3. Run Chatbot Interface
bash
streamlit run hotel_search.py
📁 Project Structure
text
hotel-data-extractor/
├── hotel_scraper.py        # Main scraper script
├── create_embeddings.py    # Creates vector embeddings
├── hotel_search.py         # Streamlit chatbot interface
├── assets.py               # Constants and configurations
├── api_management.py       # API key handling
├── llm_calls.py            # LLM interaction logic
├── markdown.py             # Markdown processing
├── utils.py                # Helper functions
├── raw_data/               # Raw markdown files
├── extracted_data/         # Extracted JSON data
├── hotel_embeddings/       # Vector embeddings
├── requirements.txt        # Dependencies
└── .env                    # API keys (not committed)
