# hotel_scraper.py
import asyncio
import json
import os
import re
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv

# Import from your existing files
from assets import GEMINI_MODEL_FULLNAME
from api_management import get_api_key
from llm_calls import call_llm_model
from crawl4ai import AsyncWebCrawler

# Load environment variables
load_dotenv()

# Create directories for storing data
os.makedirs("raw_data", exist_ok=True)
os.makedirs("extracted_data", exist_ok=True)
os.makedirs("url_lists", exist_ok=True)

# System message for hotel data extraction
HOTEL_SYSTEM_MESSAGE = """You are a specialized hotel data extraction assistant. Extract the following information from the provided hotel website content:

1. Hotel name and location (as separate fields)
2. Rooms information:
   - Room type
   - Price (look for any price information, including "From X INR/Night")
   - Availability status
   - Room features/amenities

3. General hotel amenities (facilities available to all guests)
4. Dining options with names and descriptions

Be thorough in extracting price information - look for any mentions of rates, including starting prices like "From 5,813 INR/Night".

Format the data as a clean JSON object with nested structures to properly associate details with specific rooms or facilities. If any information is not available, indicate with "Not specified".
"""

class Room(BaseModel):
    room_type: str
    price: str
    availability: str
    features: List[str]

class DiningOption(BaseModel):
    name: str
    description: str

class HotelDataModel(BaseModel):
    hotel_name: str
    location: str
    rooms: List[Room]
    general_amenities: List[str]
    dining_options: List[DiningOption]

def generate_unique_name(url: str) -> str:
    """Generate a unique name based on the URL"""
    return url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")

async def get_website_markdown_async(url: str) -> str:
    """
    Async function using crawl4ai's AsyncWebCrawler to produce the raw markdown.
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        if result.success:
            return result.markdown
        else:
            return ""

def fetch_website_markdown(url: str) -> str:
    """
    Synchronous wrapper around get_website_markdown_async().
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_website_markdown_async(url))
    finally:
        loop.close()

def read_raw_data(unique_name: str) -> str:
    """
    Read raw data from local file system
    """
    filepath = os.path.join("raw_data", f"{unique_name}.md")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def save_raw_data(unique_name: str, url: str, raw_data: str) -> None:
    """
    Save raw data to local file system
    """
    filepath = os.path.join("raw_data", f"{unique_name}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(raw_data)
    print(f"Raw data saved to {filepath}")

def extract_hotel_urls_from_markdown(markdown_content):
    """Extract hotel-related URLs from markdown content"""
    # Regex to extract all URLs from markdown links
    pattern = r"\[.*?\]\((https?://[^)]+)\)"
    urls = re.findall(pattern, markdown_content)
    
    # Filter URLs that are related to hotel details
    hotel_related_keywords = ['rooms', 'deluxe', 'premium', 'suite', 'king', 'twin', 
                             'eat-drink', 'dining', 'restaurant', 'bar', 'facilities']
    filtered_urls = [url for url in urls if any(keyword in url.lower() for keyword in hotel_related_keywords)]
    
    return filtered_urls

def process_hotel_urls(markdown_file_path):
    """Process a markdown file to extract hotel-related URLs"""
    # Read the markdown file
    with open(markdown_file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()
    
    # Extract hotel-related URLs
    hotel_urls = extract_hotel_urls_from_markdown(markdown_content)
    
    # Save URLs to a text file
    output_file = os.path.join("url_lists", os.path.splitext(os.path.basename(markdown_file_path))[0] + "_urls.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for url in hotel_urls:
            f.write(f"{url}\n")
    
    print(f"Extracted {len(hotel_urls)} hotel-related URLs to {output_file}")
    return hotel_urls

def process_all_markdown_files():
    """Process all markdown files in the raw_data directory"""
    # Get all markdown files in the raw_data directory
    markdown_files = [f for f in os.listdir("raw_data") if f.endswith(".md")]
    
    all_hotel_urls = []
    for md_file in markdown_files:
        file_path = os.path.join("raw_data", md_file)
        urls = process_hotel_urls(file_path)
        all_hotel_urls.extend(urls)
    
    # Save all unique URLs to a combined file
    unique_urls = list(set(all_hotel_urls))
    with open(os.path.join("url_lists", "all_hotel_urls.txt"), "w", encoding="utf-8") as f:
        for url in unique_urls:
            f.write(f"{url}\n")
    
    print(f"Total unique hotel URLs found: {len(unique_urls)}")
    return unique_urls

def fetch_and_store_markdowns(urls: List[str]) -> List[str]:
    """
    For each URL:
    1) Generate unique_name
    2) Check if there's already raw data saved locally
    3) If not found, fetch markdown
    4) Save to local file
    Return a list of unique_names (one per URL).
    """
    unique_names = []
    for url in urls:
        unique_name = generate_unique_name(url)
        
        # Check if we already have raw_data locally
        raw_data = read_raw_data(unique_name)
        
        if raw_data:
            print(f"Found existing data for {url} => {unique_name}")
        else:
            # Fetch markdown
            markdown = fetch_website_markdown(url)
            save_raw_data(unique_name, url, markdown)
        
        unique_names.append(unique_name)
    
    return unique_names

def extract_hotel_data(unique_name: str, model: str = GEMINI_MODEL_FULLNAME) -> Dict:
    """Extract hotel information using Gemini 1.5 Flash"""
    # Get the raw data
    raw_data = read_raw_data(unique_name)
    
    if not raw_data:
        print(f"No raw data found for {unique_name}")
        return {}
    
    # Define JSON schema for response format
    response_format = {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "hotel_name": {"type": "string"},
                "location": {"type": "string"},
                "rooms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "room_type": {"type": "string"},
                            "price": {"type": "string"},
                            "availability": {"type": "string"},
                            "features": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "general_amenities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "dining_options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
    
    # Call the LLM using your existing function
    parsed_response, token_counts, cost = call_llm_model(
        data=raw_data,
        response_format=response_format,
        model=model,
        system_message=HOTEL_SYSTEM_MESSAGE,
        extra_user_instruction="Focus on extracting room details, prices, availability, and amenities."
    )
    
    # Print token usage and cost
    print(f"Input tokens: {token_counts['input_tokens']}")
    print(f"Output tokens: {token_counts['output_tokens']}")
    print(f"Cost: ${cost:.4f}")
    
    # Save the extracted data
    save_extracted_data(unique_name, parsed_response)
    
    return parsed_response

def save_extracted_data(unique_name: str, extracted_data) -> None:
    """Save extracted data to a local JSON file"""
    # If it's a string that looks like JSON, parse it
    if isinstance(extracted_data, str):
        try:
            data_dict = json.loads(extracted_data)
        except json.JSONDecodeError:
            data_dict = {"raw_text": extracted_data}
    # Convert to dict if it's a model object
    elif hasattr(extracted_data, "dict"):
        data_dict = extracted_data.dict()
    elif hasattr(extracted_data, "__dict__"):
        data_dict = extracted_data.__dict__
    else:
        data_dict = extracted_data
    
    # Save the extracted data to a file
    filepath = os.path.join("extracted_data", f"{unique_name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"Extracted data saved to {filepath}")

async def main():
    """Main function to process hotel websites and extract data"""
    # First, fetch and store the main hotel page
    main_hotel_url = "https://www.essentiahotels.in/luxury-hotel-indore/"
    main_unique_name = generate_unique_name(main_hotel_url)
    
    # Fetch and store markdown for the main URL
    if not read_raw_data(main_unique_name):
        markdown = await get_website_markdown_async(main_hotel_url)
        if markdown:
            save_raw_data(main_unique_name, main_hotel_url, markdown)
    
    # Extract hotel URLs from the main markdown file
    hotel_urls = process_all_markdown_files()
    
    # If no URLs were found, use the main URL
    if not hotel_urls:
        hotel_urls = [main_hotel_url]
    
    # Process each URL with your hotel scraper
    results = {}
    for url in hotel_urls:
        unique_name = generate_unique_name(url)
        
        # Fetch and store markdown for this URL if not already done
        if not read_raw_data(unique_name):
            markdown = await get_website_markdown_async(url)
            if markdown:
                save_raw_data(unique_name, url, markdown)
        
        # Extract hotel data
        result = extract_hotel_data(unique_name, GEMINI_MODEL_FULLNAME)
        results[url] = result
    
    # Save all results to a single JSON file
    with open("hotel_data_results.json", "w", encoding="utf-8") as f:
        # Convert to serializable format
        serializable_results = {}
        for url, result in results.items():
            if hasattr(result, "dict"):
                serializable_results[url] = result.dict()
            elif hasattr(result, "__dict__"):
                serializable_results[url] = result.__dict__
            else:
                serializable_results[url] = result
                
        json.dump(serializable_results, f, indent=2)
    
    print(f"All results saved to hotel_data_results.json")
    return results

if __name__ == "__main__":
    # Run the scraper
    asyncio.run(main())
