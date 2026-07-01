from app.retrieval.search_engine import SearchEngine


engine = SearchEngine()

results = engine.search(
    "Java developer with leadership skills"
)

for i, product in enumerate(results, start=1):
    print(f"{i}. {product['name']}")