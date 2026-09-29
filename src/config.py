import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement (.env)
load_dotenv()

# Chemins de base du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# Assurer l'existence des répertoires
DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres LLM Groq & Embeddings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Paramètres de chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Paramètres de recherche vectorielle
TOP_K_RESULTS = 4
