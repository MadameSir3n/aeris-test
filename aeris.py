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
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.model_id = AERIS_MODEL
        
        try:
            print(f"Loading tokenizer for {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            print(f"Loading model for {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @bentoml.api()
    def predict(self, input: str, emotion: str = "calm", form: str = "human") -> AerisResponse:
        import torch
        
        messages = [
            {"role": "system", "content": AERIS_SYSTEM},
            {"role": "user",   "content": input},
        ]
        
        try:
            print(f"Processing request: {input}")
            
            # Format for Qwen chat model
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=160,
                    temperature=0.75,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                )
            
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            content = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            print(f"Generated response: {content}")
            
        except Exception as exc:
            print(f"Error during prediction: {exc}")
            content = f"Aeris is resting — {exc}"

        return AerisResponse(
            output=content,
            model=self.model_id,
            emotion=emotion,
        )

    @bentoml.api()
    def health(self) -> dict:
        return {"status": "ok", "model": self.model_id}
