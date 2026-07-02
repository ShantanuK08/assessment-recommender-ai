import json
import os

from dotenv import load_dotenv

from app.agent.conversation_analyzer import ConversationAnalyzer
from app.retrieval.search_engine import SearchEngine

load_dotenv()

analyzer = ConversationAnalyzer(
    api_key=os.getenv("GEMINI_API_KEY")
)

search = SearchEngine()

messages = [
    {
        "role": "user",
        "content": "We are hiring a Java developer with 4 years experience."
    }
]

context = analyzer.analyze(messages)

print("=" * 60)
print("SEARCH CONTEXT")
print("=" * 60)
print(json.dumps(context, indent=4))

print("\n" + "=" * 60)
print("RETRIEVED PRODUCTS")
print("=" * 60)

results = search.search(context, top_k=10)

print(f"\nRetrieved {len(results)} assessments\n")

for i, product in enumerate(results, 1):
    print(f"{i}. {product.get('name')}")
    print(f"   Type : {product.get('test_type')}")
    print(f"   URL  : {product.get('url')}")
    print()