"""Central configuration. Import from here, don't hardcode in modules."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ENTITIES_CSV = DATA_DIR / "entities.csv"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DIR = ROOT / ".chroma"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

# --- Model names ---
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
IMAGE_EMBED_MODEL = "clip-ViT-B-32"
GEMINI_MODEL = "gemini-2.5-flash"

# --- ChromaDB collections ---
TEXT_COLLECTION = "entities_text"
IMAGE_COLLECTION = "entities_image"

# --- Fusion weights (tune after first eval, document any changes) ---
# final = ALPHA*query_text + BETA*taste_align + GAMMA*meta_match + DELTA*image_sim
ALPHA = 0.50
BETA = 0.30
GAMMA = 0.15
DELTA = 0.05

# --- Retrieval ---
TOP_K_CANDIDATES = 20
TOP_K_FINAL = 5

# --- Taste vector config ---
# weight_i = (rating_i - RATING_NEUTRAL) / RATING_SCALE
# rating=10 -> +1.0;  rating=5 -> 0;  rating=1 -> -0.8
RATING_NEUTRAL = 5.0
RATING_SCALE = 5.0

# Mood sub-centroids. Must match controlled vocabulary in entities.csv.
MOOD_CENTROIDS = ["tragic", "epic", "political", "cathartic", "goofy"]
MIN_GAMES_PER_MOOD = 5
MIN_RATING_FOR_MOOD = 7

# --- Intent classifier labels ---
INTENT_LABELS = [
    "factual",
    "similarity",
    "comparative",
    "mood_tragic",
    "mood_epic",
    "mood_political",
    "mood_cathartic",
    "mood_goofy",
    "mood_general",
]

# --- Eval ---
TEST_SET_PATH = EVAL_DIR / "test_set.json"
