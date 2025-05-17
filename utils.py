# utils.py
def generate_unique_name(url: str) -> str:
    """Generate a unique name based on the URL"""
    return url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
