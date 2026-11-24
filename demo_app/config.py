import os
from pathlib import Path

# Automatically load .env from project root if python-dotenv is present
try:
    from dotenv import load_dotenv
    root_dir = Path(__file__).resolve().parent.parent
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass

def get_gemini_api_key() -> str:
    """Returns the Gemini API key from backend environment (.env or system env)."""
    return os.environ.get("GEMINI_API_KEY", "").strip()

def is_gemini_enabled() -> bool:
    """Checks whether the Gemini Generative Author is configured in backend."""
    return bool(get_gemini_api_key())
