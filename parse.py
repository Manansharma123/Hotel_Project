import json

# Load the raw JSON from the file
with open("hotel_data_results.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Initialize list for all parsed hotel entries
all_hotels = []

# Process each URL key and parse its JSON string
for url, json_string in raw_data.items():
    try:
        hotel_data = json.loads(json_string)
        hotel_data["source_url"] = url  # Add the URL to the hotel object for traceability
        all_hotels.append(hotel_data)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for URL: {url} — {e}")

# Write the cleaned hotel data to a structured JSON file
with open("cleaned_hotel_data.json", "w", encoding="utf-8") as f:
    json.dump(all_hotels, f, indent=4, ensure_ascii=False)

print("✅ Cleaned hotel data saved to 'cleaned_hotel_data.json'")
