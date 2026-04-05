from src.retrieval.colbert.indexer import rebuild_from_gold_sources, retrieve

print("Rebuilding index...")
rebuild_from_gold_sources("service_agreements")
print("Index rebuilt.")

queries = [
    "What is the agreement number?",
    "What are the compensation terms?",
    "Who is the service provider?",
]
results = retrieve("service_agreements", queries, top_k=5)
for r in results:
    c = r["content"][:120]
    s = r["score"]
    print(f"  Score: {s:.4f} | {c}...")
print(f"Retrieved {len(results)} chunks.")
