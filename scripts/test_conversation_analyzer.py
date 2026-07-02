import json
import os

from dotenv import load_dotenv

from app.agent.conversation_analyzer import ConversationAnalyzer

load_dotenv()

analyzer = ConversationAnalyzer(
    api_key=os.getenv("GEMINI_API_KEY")
)

messages = [
    {
        "role": "user",
        "content": "We are hiring a Java developer with 4 years experience."
    }
]

context = analyzer.analyze(messages)

print(json.dumps(context, indent=4))