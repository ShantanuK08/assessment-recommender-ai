#app\core\llm.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class LLMClient:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt: str):
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            error = str(e)

            if "429" in error or "quota" in error.lower():
                return (
                    "The language model is temporarily unavailable because the API "
                    "quota has been reached. Please try again in a few minutes."
                )

            raise RuntimeError(f"LLM generation failed: {e}")