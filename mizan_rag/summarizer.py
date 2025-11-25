"""
Mizan RAG - Universal Summarizer
================================

A fully general LLM summarizer with support for:

✔ OpenAI (GPT-4.x, GPT-4o, GPT-o)
✔ Groq (LLaMA-3.x SpecDec)
✔ OpenRouter (100+ models)
✔ HuggingFace (chat & text-generation)
✔ Local HF models
✔ Future Mizan-LLM models

Key Features:
-------------
- Automatic provider detection
- Unified prompt interface
- Caching support (MizanCache)
- Retry + error handling
- GPU-aware local HF inference
- Consistent behavior across all LLMs

Public API:
-----------
summarize(text)
summarize_chunks([chunk1, chunk2])
answer_question(question, chunks)

Author: Ahsan Shaokat
Commercial Version: READY
"""

import os
import torch
from typing import List, Optional

from mizan_rag.utils.cache import MizanCache

# ------------------------------------
# Optional third-party imports
# ------------------------------------
try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    from groq import Groq
except:
    Groq = None

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
    )
except:
    AutoTokenizer = None
    AutoModelForCausalLM = None


# =====================================================================
#                     UNIVERSAL SUMMARIZER CLASS
# =====================================================================

class MizanSummarizer:
    """
    Universal Summarizer for Mizan RAG.

    Supports:
        - OpenAI
        - Groq
        - OpenRouter
        - HuggingFace models
        - Local HF models

    Optionally uses MizanCache for:
        - caching generated LLM outputs
        - reducing API calls
    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        cache: Optional[MizanCache] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.cache = cache

        # Get API key from env if possible
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )

        # Auto-detect provider if not specified
        self.provider = provider or self._auto_detect_provider(model)

        # Initialize proper backend client
        if self.provider == "openai":
            self._init_openai()

        elif self.provider == "groq":
            self._init_groq()

        elif self.provider == "openrouter":
            self._init_openrouter()

        elif self.provider in {"hf", "local"}:
            self._init_huggingface(model)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")


    # =================================================================
    #                      PROVIDER AUTO-DETECTION
    # =================================================================

    def _auto_detect_provider(self, model: str) -> str:
        m = model.lower()

        if m.startswith("gpt"):
            return "openai"

        if "specdec" in m or "groq" in m:
            return "groq"

        if os.path.isdir(model):
            return "local"

        if any(x in m for x in ["llama", "mistral", "qwen", "starcoder", "falcon"]):
            return "hf"

        if self.base_url:
            return "openrouter"

        return "hf"


    # =================================================================
    #                      PROVIDER INIT
    # =================================================================

    def _init_openai(self):
        if OpenAI is None:
            raise ImportError("Install: pip install openai")

        self.client = OpenAI(api_key=self.api_key)

    def _init_groq(self):
        if Groq is None:
            raise ImportError("Install: pip install groq")

        self.client = Groq(api_key=self.api_key)

    def _init_openrouter(self):
        if OpenAI is None:
            raise ImportError("Install: pip install openai")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _init_huggingface(self, model: str):
        if AutoTokenizer is None:
            raise ImportError("Install: pip install transformers accelerate")

        print(f"[HF] Loading local or HF model: {model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True
        )

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        self.has_template = hasattr(self.tokenizer, "apply_chat_template")



    # =================================================================
    #                      INTERNAL LLM INVOKE
    # =================================================================

    def _invoke_llm(self, prompt: str) -> str:

        # ---- Cache lookup ----
        if self.cache:
            cached = self.cache.get_response(prompt)
            if cached:
                return cached

        # ---- OpenAI / Groq / OpenRouter ----
        if self.provider in ["openai", "groq", "openrouter"]:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            result = resp.choices[0].message.content.strip()

        # ---- HuggingFace / Local ----
        elif self.provider in ["hf", "local"]:

            messages = [
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": prompt},
            ]

            if self.has_template:
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False
                )
            else:
                input_text = prompt

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt"
            ).to(self.hf_model.device)

            output = self.hf_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )

            result = self.tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

        else:
            raise RuntimeError("Invalid provider in LLM invocation.")

        # ---- Save to cache ----
        if self.cache:
            self.cache.store_response(prompt, result)

        return result


    # =================================================================
    #                          PUBLIC METHODS
    # =================================================================

    def summarize(self, text: str, max_words: int = 150) -> str:
        prompt = f"""
Summarize the following text in **{max_words} words**.

Content:
{text}

Summary:
        """
        return self._invoke_llm(prompt)

    def summarize_chunks(self, chunks: List[str], max_words: int = 250) -> str:
        joined = "\n\n".join(chunks)

        prompt = f"""
    You are a professional summarizer working inside the Mizan RAG Pipeline.

    Your job is to create a single, coherent summary of ALL the provided chunks.

    Rules:
    - Use information from ALL chunks collectively.
    - Merge overlapping ideas; avoid repetition.
    - No assumptions outside the text.
    - No hallucinations.
    - If information is incomplete or fragmented, summarize what *is* present.
    - Focus on factual accuracy and clarity.

    Output limit: {max_words} words.

    Chunks:
    {joined}

    Summary:
        """

        return self._invoke_llm(prompt)

    def answer_question(self, question: str, chunks: List[str]) -> str:
        joined = "\n\n".join(chunks)

        print(joined)
        prompt = f"""
    You are an evidence-based reasoning engine inside the Mizan RAG Pipeline.

    Your task is to answer the question using ONLY the information found in the provided chunks.

    Reasoning Rules:
    1. Use *all* chunks together; combine them when they refer to related events.
    2. If the chunks contain partial clues, produce a partial but correct answer.
    3. Do NOT hallucinate or add facts that are not present.
    4. If the chunks contain no relevant information, respond exactly with:
    "There is not enough information in the retrieved chunks to answer this question."
    5. You may infer high-level meaning from multiple scenes if evidence supports it.
    6. Be concise, factual, and focused only on what the chunks allow.

    Question:
    {question}

    Chunks:
    {joined}

    Final Answer:
        """

        return self._invoke_llm(prompt)

