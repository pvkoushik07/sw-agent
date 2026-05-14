"""Gemini wrapper. Three functions: classify_intent, synthesise_answer, judge_groundedness.
All have JSON validation and fallbacks so the agent never crashes on a bad LLM response.
"""
from __future__ import annotations

import json
import os
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from . import config

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel(config.GEMINI_MODEL)


# ----------------------------------------------------------------------
# Intent classification
# ----------------------------------------------------------------------

_INTENT_PROMPT = """You are a query intent classifier for a personal Star Wars retrieval system.

Classify the user's query into EXACTLY ONE of these labels:

- factual: Direct lookup of a specific stored fact about a named entity, or simple metadata filter.
    Examples: "what did I rate Andor?", "which Sith are in my catalogue?",
              "ships from the original trilogy"

- similarity: User describes a concrete visual or named anchor and wants matching entities.
  Personal taste should NOT bias these.
    Examples: "the character with black armor and red lightsaber",
              "small green creature with long ears",
              "which planet has red salt under a white crust"

- comparative: Multi-attribute filtering, ranking, or cross-membership queries.
    Examples: "Jedi I rated 9+ from the prequel era",
              "characters in both Rogue One and the original trilogy",
              "which prequel-era ship did I rate highest"

- mood_tragic: Subjective query seeking sad, doomed, fallen, or grief-laden content.
    Examples: "a deeply tragic character arc",
              "something that will make me feel something sad"

- mood_epic: Subjective query seeking large-scale, heroic, mythic content.
    Examples: "something epic for a rewatch",
              "a sweeping Star Wars moment"

- mood_political: Subjective query seeking intrigue, ideology, scheming, institutional drama.
    Examples: "political intrigue Star Wars", "scheming villains and senate plots"

- mood_cathartic: Subjective query seeking payoff, vindication, release moments.
    Examples: "something cathartic that pays off a long setup",
              "a satisfying climactic moment"

- mood_goofy: Subjective query seeking fun, lighter, comic-relief content.
    Examples: "something fun and lighter for casual watching",
              "comic relief Star Wars", "the silly stuff"

- mood_general: Subjective with no clear specific mood signal.
    Examples: "recommend something I'd love", "surprise me"

Respond with ONLY a JSON object: {"intent": "<label>", "confidence": <0-1 float>}.
No other text, no markdown fences.

Query: __QUERY__
"""


def classify_intent(query: str) -> dict[str, Any]:
    """Return {"intent": str, "confidence": float}. Defaults to mood_general on failure."""
    prompt = _INTENT_PROMPT.replace("__QUERY__", query)
    try:
        resp = _model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        data = json.loads(resp.text)
        intent = data.get("intent", "mood_general")
        if intent not in config.INTENT_LABELS:
            intent = "mood_general"
        return {
            "intent": intent,
            "confidence": float(data.get("confidence", 0.5)),
        }
    except Exception as e:
        print(f"[llm] classify_intent failed: {e}. Defaulting to mood_general.")
        return {"intent": "mood_general", "confidence": 0.0}


# ----------------------------------------------------------------------
# Answer synthesis
# ----------------------------------------------------------------------

_SYNTH_PROMPT = """You are a personal Star Wars recommendation assistant for the user.

The user asked: "__QUERY__"

Here are the top retrieved entries from their personal catalogue. Use ONLY this
information — do NOT invent facts about entities not listed here.

__CANDIDATES__

Write a SHORT response (2-4 sentences) that:
1. Directly answers the question or makes a recommendation.
2. Names the most relevant entity (entities) and briefly says why they fit.
3. Speaks in second person ("you rated", "you said").
4. Quotes a short phrase from the user's take when it strengthens the answer.
5. Does NOT hallucinate any rating, era, or detail not in the data above.

Response:
"""


def synthesise_answer(query: str, candidates: list[dict]) -> str:
    """Generate a grounded answer from retrieved candidates."""
    if not candidates:
        return "I couldn't find anything in your catalogue that matches."

    formatted = []
    for i, c in enumerate(candidates, 1):
        formatted.append(
            f"{i}. {c.get('name')} ({c.get('first_appearance')}) — "
            f"rated {c.get('your_rating')}/10, type: {c.get('type')}, "
            f"era: {c.get('era')}, mood: {c.get('mood')}.\n"
            f"   Your take: {c.get('your_take')}"
        )
    candidates_str = "\n".join(formatted)

    prompt = _SYNTH_PROMPT.replace("__QUERY__", query).replace(
        "__CANDIDATES__", candidates_str
    )
    try:
        resp = _model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"[llm] synthesise_answer failed: {e}. Falling back to template.")
        top = candidates[0]
        return (
            f"Based on your catalogue, **{top['name']}** seems like a strong fit "
            f"(you rated it {top['your_rating']}/10). {top['your_take']}"
        )


# ----------------------------------------------------------------------
# LLM-as-judge (used only in eval)
# ----------------------------------------------------------------------

_JUDGE_PROMPT = """Score the following answer for groundedness on a scale of 1-5.

5 = Every claim is supported by the provided context.
3 = Mostly grounded, some unsupported details.
1 = Hallucinated facts not in context.

Query: __QUERY__
Context (retrieved entities): __CONTEXT__
Answer: __ANSWER__

Respond with ONLY a JSON object: {"score": <1-5 int>, "reasoning": "<one sentence>"}
"""


def judge_groundedness(query: str, context: str, answer: str) -> dict[str, Any]:
    """Used by eval/evaluate.py. Returns {"score": int, "reasoning": str}."""
    prompt = (
        _JUDGE_PROMPT.replace("__QUERY__", query)
        .replace("__CONTEXT__", context)
        .replace("__ANSWER__", answer)
    )
    try:
        resp = _model.generate_content(
            prompt, generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(resp.text)
        return {"score": int(data.get("score", 3)), "reasoning": data.get("reasoning", "")}
    except Exception as e:
        print(f"[llm] judge_groundedness failed: {e}.")
        return {"score": 0, "reasoning": "judge_failed"}
