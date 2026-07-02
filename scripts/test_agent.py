import json

from app.agent.conversation_agent import ConversationAgent

agent = ConversationAgent()

messages = [
    {
        "role": "user",
        "content": "We are hiring a Java Developer with 4 years experience."
    }
]

response = agent.chat(messages)

print(json.dumps(response, indent=4))