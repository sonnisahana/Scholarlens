"""Central configuration. Override anything via environment variables or a .env file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA = ROOT / "data"
RAW = DATA / "raw_pdfs"
PROCESSED = DATA / "processed"
RULES_DIR = DATA / "rules"
REGISTRY = DATA / "schemes_registry.csv"
GOLD_PERSONAS = DATA / "gold_set" / "personas.jsonl"
GOLD_RETRIEVAL = DATA / "gold_set" / "retrieval_queries.jsonl"
CHUNKS = PROCESSED / "chunks.jsonl"
SENTENCES = PROCESSED / "sentences.jsonl"
RESULTS = ROOT / "results"
CACHE_DIR = ROOT / ".cache" / "llm"

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
NLI_THRESHOLD = float(os.getenv("NLI_THRESHOLD", "0.5"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
LLM_CACHE = os.getenv("LLM_CACHE", "1") == "1"   # cache temperature-0 calls on disk

CHUNK_MAX_SENT = 6
CHUNK_OVERLAP = 1
TOP_K = 8
RETRIEVAL_POOL = 30
