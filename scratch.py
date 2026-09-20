import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, val = line.split('=', 1)
                os.environ[key] = val

from src.embeddings import GeminiEmbedder
from src.chunking import compute_similarity

# Let GeminiEmbedder use its default or ENV model (gemini-embedding-001)
embedder = GeminiEmbedder()

pairs = [
    ("hello world", "hello world"),
    ("Thời gian hoàn trả là bao lâu?", "Bao nhiêu ngày thì được trả hàng?"),
    ("Tôi thích ăn táo.", "Apple vừa ra mắt điện thoại."),
    ("Chính sách đổi trả hàng.", "Chính sách đổi trả hàng!"),
    ("Mèo thích ăn cá.", "Hệ điều hành Windows.")
]

for i, (a, b) in enumerate(pairs, 1):
    sim = compute_similarity(embedder(a), embedder(b))
    print(f"Pair {i}: {sim:.4f}")
