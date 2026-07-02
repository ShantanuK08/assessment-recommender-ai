"""
app/agent/conversation_agent.py

ConversationAgent is the ONLY orchestrator ("main brain") in this system.

Locked architecture:

    ConversationAgent
            |
    ConversationAnalyzer   (conversation understanding only)
            |
    Structured Search Context
            |
    Python decision logic         <-- lives here, in ConversationAgent
            |
    SearchEngine            (retrieval only)
            |
    LLMClient               (natural language generation only)
            |
    Final JSON response

ConversationAgent is responsible for:
    - Calling ConversationAnalyzer to get the Search Context.
    - Deciding, in Python, which of the five actions applies:
          clarify | recommend | refine | compare | refuse
    - Calling SearchEngine for retrieval when needed.
    - Calling LLMClient for the natural-language reply.
    - Assembling the final API response.

It does NOT understand raw conversation text itself (that's the
Analyzer's job), does NOT perform retrieval itself (that's the
SearchEngine's job), and does NOT generate language itself (that's
the LLMClient's job).
"""

from typing import Any, Dict, List

from app.agent.conversation_analyzer import ConversationAnalyzer
from app.retrieval.search_engine import SearchEngine
from app.core.llm import LLMClient


class ConversationAgent:

    def __init__(self):
        self.analyzer = ConversationAnalyzer()
        self.search = SearchEngine()
        self.llm = LLMClient()

    # ------------------------------------------------------------------
    # Public entry point (preserved signature)
    # ------------------------------------------------------------------

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        messages = [{"role": "user"|"assistant", "content": "..."}, ...]

        Stateless entry point: the full conversation is reconstructed
        into a Search Context on every call, a routing decision is made
        in Python, and the appropriate handler produces the final response.
        """
        search_context = self.analyzer.analyze(messages)
        action = self._decide_action(search_context, messages)

        if action == "refuse":
            return self._handle_refuse()

        if action == "clarify":
            return self._handle_clarify(search_context)

        if action == "compare":
            return self._handle_compare(search_context)

        # "recommend" and "refine" share the same retrieval + reply path;
        # only the framing given to the LLM differs.
        return self._handle_recommend(
            search_context, messages, is_refine=(action == "refine")
        )

    # ------------------------------------------------------------------
    # Python decision logic (the only place routing decisions are made)
    # ------------------------------------------------------------------

    def _decide_action(
        self, search_context: Dict[str, Any], messages: List[Dict[str, str]]
    ) -> str:
        """
        Determines the workflow action from the Search Context.
        This is a deterministic Python decision — the Analyzer's
        `intent` field is only ever used as a hint, never as the
        final answer.
        """
        if search_context["intent"] == "off_topic":
            return "refuse"

        if len(search_context["comparison_items"]) >= 2:
            return "compare"

        if not self._has_enough_signal(search_context):
            return "clarify"

        if search_context["intent"] == "refine" and self._has_prior_recommendation_turn(messages):
            return "refine"

        return "recommend"

    @staticmethod
    def _has_enough_signal(search_context: Dict[str, Any]) -> bool:
        """True if there's enough information to attempt a recommendation."""
        return bool(
            search_context["role"]
            or search_context["keys"]
            or search_context["assessment_preferences"]
        )

    @staticmethod
    def _has_prior_recommendation_turn(messages: List[Dict[str, str]]) -> bool:
        """
        True if the conversation already went through at least one
        recommend/refine cycle, i.e. this isn't the user's first ask.
        """
        user_turns = [m for m in messages if m.get("role") == "user"]
        return len(user_turns) > 1

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_clarify(self, search_context: Dict[str, Any]) -> Dict[str, Any]:
        """Not enough signal yet — ask one targeted clarifying question."""
        missing = search_context["missing_fields"] or ["role", "required skills"]

        prompt = f"""
You are an SHL Assessment Recommendation Assistant.
The user's request is too vague to recommend assessments confidently.

Missing information: {missing}

Ask ONE short, specific clarifying question to get the most important
missing piece of information. Do not recommend anything yet.
"""
        reply = self.llm.generate(prompt)
        return self._build_response(reply, recommendations=[])

    def _handle_refuse(self) -> Dict[str, Any]:
        """Latest message is unrelated to SHL assessments/hiring."""
        prompt = """
The user's latest message is unrelated to SHL assessments or hiring.
Politely explain that you can only help with SHL assessment
recommendations, and invite them to ask about hiring assessments instead.
"""
        reply = self.llm.generate(prompt)
        return self._build_response(reply, recommendations=[])

    def _handle_recommend(
        self,
        search_context: Dict[str, Any],
        messages: List[Dict[str, str]],
        is_refine: bool,
    ) -> Dict[str, Any]:
        """Fresh recommendation or refinement of a prior one — same retrieval path."""
        products = self.search.search(search_context, top_k=10)
        catalog_context = self._format_products_for_prompt(products)

        framing = (
            "The user is refining their earlier request. Acknowledge what "
            "changed and present an UPDATED set of recommendations rather "
            "than starting over."
            if is_refine
            else
            "Present these recommendations clearly and briefly explain why "
            "each fits the stated role/criteria."
        )

        prompt = f"""
You are an SHL Assessment Recommendation Assistant.
Only reference assessments from the catalog context below — never invent one.

{framing}

Search Context:
{search_context}

Catalog Context:
{catalog_context}

Conversation:
{messages}
"""
        reply = self.llm.generate(prompt)
        return self._build_response(reply, self._to_recommendation_objects(products))

    def _handle_compare(self, search_context: Dict[str, Any]) -> Dict[str, Any]:
        """User asked to compare two or more named assessments."""
        products = self.search.find_by_names(search_context["comparison_items"])
        catalog_context = self._format_products_for_prompt(products)

        prompt = f"""
You are an SHL Assessment Recommendation Assistant.
Compare ONLY the assessments below, using ONLY the information given.
Do not use outside knowledge. If information for a field is not present,
say it's not specified rather than guessing.

Assessments to compare:
{catalog_context}
"""
        reply = self.llm.generate(prompt)
        return self._build_response(reply, self._to_recommendation_objects(products))

    # ------------------------------------------------------------------
    # Private formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_products_for_prompt(products: List[Dict[str, Any]]) -> str:
        """Renders retrieved product objects into plain text for the LLM prompt."""
        blocks = []
        for product in products:
            blocks.append(
                f"Name: {product.get('name')}\n"
                f"Test Type: {product.get('test_type', 'Unknown')}\n"
                f"Description: {product.get('description', '')}\n"
                f"URL: {product.get('url')}\n"
            )
        return "\n".join(blocks)

    @staticmethod
    def _to_recommendation_objects(products: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Maps retrieved product objects onto the required response schema."""
        return [
            {
                "name": product.get("name"),
                "url": product.get("url"),
                "test_type": product.get("test_type", "Unknown"),
            }
            for product in products
        ]

    @staticmethod
    def _build_response(
        reply: str, recommendations: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Assembles the final API response in the exact required schema."""
        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": False,
        }