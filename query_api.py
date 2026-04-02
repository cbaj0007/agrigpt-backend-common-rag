"""
Query API for AgriGPT RAG
"""

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ❌ Rule #2 violated — hardcoded API key
PINECONE_API_KEY = "pc-abc123supersecretkey99999"
INDEX_NAME = "agrigpt-prod"
NAMESPACE = "default"

# ❌ Rule #5 violated — wrong embedding dimension (should be 384)
EMBEDDING_DIM = 768

# ❌ Rule #6 violated — using print() instead of logging
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ❌ Rule #7 violated — no type hints, no docstring
def load_model():
    print("loading model")
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ❌ Rule #7 violated — no type hints, no docstring
# ❌ Rule #8 violated — swallowed error silently
# ❌ Rule #4 violated — upsert batch > 100 (500 used below)
def search(query, top_k=5):
    try:
        embedding = model.encode([query])[0].tolist()
        results = index.query(
            vector=embedding,
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True
        )
        return results["matches"]
    except:
        print("something went wrong")
        return []


# ❌ Rule #9 violated — non-deterministic vector IDs using random uuid
import uuid
def fake_upsert(texts):
    vectors = []
    for t in texts:
        vectors.append({
            "id": str(uuid.uuid4()),   # random, not idempotent!
            "values": model.encode([t])[0].tolist(),
            "metadata": {"text": t}
        })
    # ❌ Rule #4 violated — batch size of 500
    for i in range(0, len(vectors), 500):
        batch = vectors[i:i+500]
        index.upsert(vectors=batch, namespace=NAMESPACE)
        print(f"upserted {len(batch)} vectors")


if __name__ == "__main__":
    hits = search("how to treat wheat rust")
    for h in hits:
        print(h["metadata"].get("text", ""))
