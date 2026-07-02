"""
app/agent/conversation_analyzer.py

Single responsibility: understand the conversation and extract a
structured Search Context. This module NEVER:
    - searches the catalog
    - decides the final action (clarify/recommend/refine/compare/refuse)
    - generates the user-facing reply

It ONLY turns raw conversation history into a structured object that
SearchEngine and ConversationAgent can consume deterministically.

Why re-extract from full history every call?
    The API is stateless. Every /chat request receives the *entire*
    conversation, so on every turn we reconstruct the cumulative Search
    Context from scratch by giving Gemini the full transcript. Anything
    stated earlier (role, keys, preferences) naturally carries forward
    as long as it's still present in `messages`.
"""

import json
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai


# Every field we always want present, so downstream code never has to
# defensively .get() with a dozen different defaults.
_SEARCH_CONTEXT_TEMPLATE: Dict[str, Any] = {
    "intent": "recommend",       # hint only — Python makes the final call
    "role": "",
    "experience": "",
    "job_level": "",
    "keys": [],
    "remote": None,
    "adaptive": None,
    "languages": [],
    "assessment_preferences": [],
    "comparison_items": [],
    "missing_fields": [],
}

_ALLOWED_INTENTS = {"recommend", "refine", "compare", "off_topic", "chit_chat"}


class ConversationAnalyzer:
    """
    Turns `messages` into a structured Search Context using Gemini for
    language understanding only. No search, no recommendations, no
    final routing decision — that stays in ConversationAgent.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def analyze(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        prompt = self._build_prompt(messages)
        raw = self.model.generate_content(prompt).text.strip()
        raw = self._strip_code_fences(raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fail safe: if Gemini returns malformed JSON, don't crash the
            # request — fall back to an empty context. The agent will then
            # naturally route to "clarify" since no signal is present.
            parsed = {}

        return self._normalize(parsed)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return f"""
You are a language-understanding module for an SHL assessment recommender.

Your ONLY job is to read the full conversation and extract a structured
"Search Context" JSON object. You do not recommend assessments. You do not
decide what the system should do next beyond a rough `intent` hint.

Return STRICT JSON with exactly these fields:

{{
  "intent": one of {sorted(_ALLOWED_INTENTS)},
  "role": string (job role/title being hired for, "" if unknown),
  "experience": string (e.g. "entry", "mid", "senior", "" if unknown),
  "job_level": string (e.g. "Individual Contributor", "Manager", "" if unknown),
  "keys": array of short skill/competency keywords (e.g. "Java", "Leadership"),
  "remote": true | false | null (null if not mentioned),
  "adaptive": true | false | null (null if not mentioned),
  "languages": array of language names required for the assessment content,
  "assessment_preferences": array of assessment TYPES the user explicitly
      wants (e.g. "Personality", "Cognitive", "Situational Judgement"),
  "comparison_items": array of specific assessment NAMES the user wants
      compared against each other (only when the user is asking for a
      comparison between named assessments),
  "missing_fields": array of field names from the above that are still
      unknown and would meaningfully improve recommendation quality
      (e.g. ["role"] if we still don't know who they're hiring for)
}}

Rules:
- Accumulate information across the ENTIRE conversation, not just the
  latest message. If role was given two turns ago, keep it.
- Use intent="off_topic" ONLY if the latest user message is unrelated to
  hiring/assessments (e.g. asking about labour law, weather, etc).
- Use intent="compare" only when comparison_items has 2+ named assessments.
- Use intent="refine" when the user is adjusting/adding to criteria they
  already gave earlier in the conversation.
- Use intent="recommend" for a fresh or first-time recommendation request.
- Never invent role/keys/preferences that were not stated or clearly implied.
- Return ONLY the JSON object. No markdown fences, no commentary.

Conversation:
{json.dumps(messages, indent=2)}
"""

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "")
        return text.strip()

    @staticmethod
    def _normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
        context = dict(_SEARCH_CONTEXT_TEMPLATE)
        context.update({k: v for k, v in parsed.items() if k in context})

        if context["intent"] not in _ALLOWED_INTENTS:
            context["intent"] = "recommend"

        for list_field in ("keys", "languages", "assessment_preferences",
                            "comparison_items", "missing_fields"):
            if not isinstance(context[list_field], list):
                context[list_field] = []

        return context