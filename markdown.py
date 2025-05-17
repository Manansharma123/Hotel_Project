# markdown.py
import asyncio
import os
from typing import List
from utils import generate_unique_name
from crawl4ai import AsyncWebCrawler

# Create directories for storing data
os.makedirs("raw_data", exist_ok=True)

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
