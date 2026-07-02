from app.core.llm import LLMClient

llm = LLMClient()

response = llm.generate("Say only: Gemini is working!")

print(response)