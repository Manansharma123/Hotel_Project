import os
from dotenv import load_dotenv
from assets import MODELS_USED

load_dotenv()

def get_api_key(model):
    """
    Returns an API key for a given model by:
    1) Looking up the environment var name in MODELS_USED[model].
    2) Returning the key from os.environ.
    """
    env_var_name = list(MODELS_USED[model])[0]  # e.g., "GEMINI_API_KEY"
    return os.getenv(env_var_name)
