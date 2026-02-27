import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
AI_PROVIDER = os.getenv("AI_PROVIDER", "openrouter").lower()   # openai | gemini | anthropic | openrouter
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

# Default models per provider if AI_MODEL is not set
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-1.5-pro",
    "anthropic": "claude-3-5-sonnet-20241022",
    # Best vision + reasoning model on OpenRouter for exam/study use
    "openrouter": "google/gemini-2.0-flash-001",
}

def get_model() -> str:
    return AI_MODEL or DEFAULT_MODELS.get(AI_PROVIDER, "google/gemini-2.0-flash-001")
