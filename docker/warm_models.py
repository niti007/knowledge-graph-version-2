"""Build-time model download.

Runs once inside `docker build` so the image already holds the embedding model,
the reranker and the spaCy pipeline. Without this, every cold start of the
Space would pull ~500 MB from the Hub before it could answer a question.

The cache location is HF_HOME (set in the Dockerfile); the same env var is set
at runtime, so sentence-transformers finds the files without going online.
"""
import os
import time

t = time.time()
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

emb = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
rer = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")

m = SentenceTransformer(emb, device="cpu")
v = m.encode(["warm"], normalize_embeddings=True)
print(f"embedding  {emb}: dim={v.shape[1]}")

ce = CrossEncoder(rer, device="cpu")
s = ce.predict([("warm", "warm")])
print(f"reranker   {rer}: score={float(s[0]):.3f}")

import spacy  # noqa: E402
nlp = spacy.load("en_core_web_sm")
print(f"spacy      en_core_web_sm {nlp.meta['version']}: {len(nlp('warm'))} token")
print(f"models ready in {time.time() - t:.1f}s; cache at {os.environ.get('HF_HOME')}")
