import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

# Load environment variables
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env")

# Ensure required directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB Settings
CHROMA_PERSIST_DIRECTORY = str(os.getenv("CHROMA_PERSIST_DIRECTORY", CHROMA_DIR))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "academic_papers")

# Embedding Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Chunking Settings
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 800))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", 100))

# Retrieval & LLM Settings
# Cheap pre-filter threshold calibrated from scripts/calibrate_threshold.py:
# filters obvious negatives before calling the LLM while letting borderline/covered
# questions reach the LLM for definitive refusal decisions.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.25))
LLM_CALL_DELAY_SECONDS = float(os.getenv("LLM_CALL_DELAY_SECONDS", 2.0))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", 3000))

# Groq LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_MAIN = os.getenv("GROQ_MODEL_MAIN", "openai/gpt-oss-120b")
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")

# Cache Settings
CACHE_VERSION = int(os.getenv("CACHE_VERSION", 2))

# Orchestration & Router Settings
MAX_QA_RETRIES = int(os.getenv("MAX_QA_RETRIES", 1))
ROUTER_HISTORY_TURNS = int(os.getenv("ROUTER_HISTORY_TURNS", 4))
ROUTER_MIN_CONFIDENCE = float(os.getenv("ROUTER_MIN_CONFIDENCE", 0.5))

# Database & Authentication Configuration
default_db_path = (DATA_DIR / "app.db").resolve()
raw_db_url = os.getenv("DATABASE_URL", "")
if not raw_db_url:
    DATABASE_URL = f"sqlite:///{default_db_path}"
elif raw_db_url.startswith("sqlite:///data/"):
    DATABASE_URL = f"sqlite:///{DATA_DIR / raw_db_url.split('sqlite:///data/')[1]}"
elif raw_db_url.startswith("sqlite:///backend/data/"):
    DATABASE_URL = f"sqlite:///{DATA_DIR / raw_db_url.split('sqlite:///backend/data/')[1]}"
else:
    DATABASE_URL = raw_db_url

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 60))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 20))
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",") if origin.strip()]
CHAT_HISTORY_LOAD_TURNS = int(os.getenv("CHAT_HISTORY_LOAD_TURNS", 4))


