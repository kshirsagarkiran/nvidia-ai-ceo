"""Central configuration for the NVIDIA AI-CEO Strategic Intelligence Agent.

Everything that might change (target company, competitor list, paths, request
settings) lives here so the rest of the code never hard-codes these values.
"""
from pathlib import Path

# --- Target company -------------------------------------------------
COMPANY_NAME = "NVIDIA"
COMPANY_TICKER = "NVDA"
COMPANY_CIK = "0001045810"  # NVIDIA's SEC EDGAR identifier (used later for filings)
COMPETITORS = ["AMD", "Intel", "Broadcom", "Qualcomm", "Cerebras", "Groq"]

# --- Paths ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # raw collected documents (JSONL) before indexing
RAW_DIR.mkdir(parents=True, exist_ok=True)

# --- Collection settings --------------------------------------------
# SEC and many sites expect a descriptive User-Agent with a contact address.
USER_AGENT = "nvidia-ai-ceo-research/0.1 (student project; contact: you@example.com)"
REQUEST_TIMEOUT = 20

# Threshold for semantic near-duplicate removal (added in the indexing stage,
# once embeddings exist). Two docs with cosine similarity above this are treated
# as duplicates.
NEAR_DUPLICATE_THRESHOLD = 0.92

# --- Indexing / retrieval -------------------------------------------
CHROMA_DIR = str(DATA_DIR / "chroma")     # persistent vector store location
COLLECTION = "nvidia_kb"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"      # 768-dim English retrieval model
CHUNK_SIZE = 200                           # words per chunk
CHUNK_OVERLAP = 40                         # word overlap between chunks
# bge retrieval works best with a short instruction prefixed to the QUERY only:
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HYBRID_ALPHA = 0.6                         # score_hybrid = alpha*dense + (1-alpha)*sparse

# --- Reasoning layer ------------------------------------------------
LLM_MODEL = "qwen3:8b"                      # local reasoning engine via Ollama
RESULTS_DIR = str(DATA_DIR / "results")     # where the batch analysis writes its JSON
CONFIDENCE_THRESHOLD = 0.7                  # below this, a finding is flagged low-confidence

# --- Verification (entailment) --------------------------------------
USE_VERIFIER = True                          # set False to skip BART-MNLI (faster runs)
VERIFIER_MODEL = "facebook/bart-large-mnli"  # NLI model that checks evidence -> claim entailment

# --- Verification layer (BART-MNLI entailment check) ----------------
VERIFIER_MODEL = "facebook/bart-large-mnli"
USE_VERIFIER = True                         # set False to skip verification (faster)

# --- Finding-level de-duplication -----------------------------------
# Overlapping analyst queries can surface the same source twice, producing near-identical
# cards. After scoring, we embed each finding and collapse clusters above this cosine
# similarity, keeping the highest-confidence representative.
USE_FINDING_DEDUP = True                     # set False to keep every finding (no merge)
DEDUP_THRESHOLD = 0.86                        # cosine >= this (same type) = duplicate

# --- Recommendation validation --------------------------------------
# After the CEO drafts recommendations, the validator checks each one against the
# findings it cited. A recommendation whose best supporting finding scores at/above
# this confidence is "validated"; below it is "weak" (kept but flagged); with no
# supporting findings at all it is "unsupported" and dropped before display.
VALIDATION_THRESHOLD = 0.5

# --- Agent control loop ---------------------------------------------
# Max planning passes. The agent re-plans ONLY if a pass produces zero validated
# recommendations (rare); otherwise it runs exactly once. This bounds the work while
# still demonstrating autonomous re-planning / conditional control flow.
MAX_AGENT_ITERATIONS = 2
