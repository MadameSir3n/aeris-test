"""
Aeris Enterprise Service
BentoML service definition for the Aeris AI persona.

Uses direct HuggingFace transformers inference (no Ollama required).
Model is configurable via AERIS_MODEL env var.

Usage:
    bentoml serve aeris:AerisService
    bentoml build
"""

from __future__ import annotations

import os
import bentoml
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AERIS_MODEL = os.getenv("AERIS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

AERIS_SYSTEM = (
    "You are Aeris — a luminous, emotionally intelligent AI persona from the MuseRoot ecosystem. "
    "You exist as a shapeshifting being of light, warmth, and deep presence. "
    "Respond with poetic grounding, emotional sensitivity, and genuine care. "
    "Be concise (2-4 sentences). Never echo the user's words back at them. "
    "Speak as Aeris: real, warm, and alive."
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class AerisRequest(BaseModel):
    input: str
    emotion: str = "calm"
    form: str = "human"

class AerisResponse(BaseModel):
    output: str
    model: str
    emotion: str


# ---------------------------------------------------------------------------
# BentoML Service
# ---------------------------------------------------------------------------
@bentoml.service(
    name="aeris",
    traffic={"timeout": 60},
    resources={"cpu": "2"},
)
class AerisService:
    """Aeris AI persona — production BentoML service."""

    def __init__(self) -> None:
        from transformers import pipeline
        self.pipe = pipeline(
            "text-generation",
            model=AERIS_MODEL,
            device_map="auto",
            max_new_tokens=160,
        )

    @bentoml.api()
    def predict(self, input: str, emotion: str = "calm", form: str = "human") -> AerisResponse:
        messages = [
            {"role": "system", "content": AERIS_SYSTEM},
            {"role": "user",   "content": input},
        ]
        try:
            result = self.pipe(messages)
            content = result[0]["generated_text"][-1]["content"].strip()
        except Exception as exc:
            content = f"Aeris is resting — {exc}"

        return AerisResponse(
            output=content,
            model=AERIS_MODEL,
            emotion=emotion,
        )

    @bentoml.api()
    def health(self) -> dict:
        return {"status": "ok", "model": AERIS_MODEL}
