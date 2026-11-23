"""LLM access. Anthropic via langchain-anthropic when configured; deterministic offline fallback otherwise."""
from __future__ import annotations

import os
import re
from typing import Protocol


class LLM(Protocol):
    name: str
    def invoke(self, system: str, user: str, max_tokens: int = 1024) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str):
        from langchain_anthropic import ChatAnthropic
        self.name = f"anthropic:{model}"
        self._chat = ChatAnthropic(model=model, max_tokens=1024, timeout=60, max_retries=2)

    def invoke(self, system: str, user: str, max_tokens: int = 1024) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        out = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)], max_tokens=max_tokens)
        c = out.content
        return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


_STOP = set("the a an of and or to in on for is are was were be by with as at from that this it its what which who how "
            "does do under must shall any all if then than".split())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2}


class OfflineLLM:
    """Extractive, citation-preserving stand-in so the pipeline runs without network/API keys.

    It never invents facts: answers are sentences copied from the numbered sources.
    """
    name = "offline-extractive"

    def invoke(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if "HYDE_PASSAGE" in user:
            return self._hyde(user)
        if "EXTRACT_TRIPLES" in user:
            return "[]"  # signals rule-based extraction
        return self._answer(user)

    @staticmethod
    def _hyde(user: str) -> str:
        q = re.search(r"QUESTION:\s*(.*?)\s*ANGLE:\s*(.*)", user, re.S)
        question, angle = (q.group(1), q.group(2).strip()) if q else (user, "")
        return (f"{angle}: Under the applicable statute and notification, {question.rstrip('?')}. "
                f"The provision prescribes the obligation, the authority responsible, the applicable section or rule, "
                f"and the consequence or penalty for non-compliance.")

    @staticmethod
    def _answer(user: str) -> str:
        qm = re.search(r"QUESTION:\s*(.*?)\s*(?:<context>|$)", user, re.S)
        question = qm.group(1) if qm else ""
        qw = _words(question)
        sources = re.findall(r"\[S(\d+)\] \(.*?\)\n(.*?)(?=\n\n\[S\d+\]|\n</context>)", user, re.S)
        scored = []
        for sid, text in sources:
            for sent in re.split(r"(?<=[.;])\s+|\n+", text):
                s = sent.strip()
                if len(s) > 25:
                    scored.append((len(qw & _words(s)), int(sid), s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [t for t in scored if t[0] > 0][:4]
        if not top:
            return ("## Answer\nThe retrieved sources do not contain sufficient information.\n\n"
                    "## Gaps & Limitations\n- No relevant passage was retrieved.")
        body = "\n".join(f"- {s} [S{sid}]" for _, sid, s in top)
        return f"## Answer\n{body}\n\n## Gaps & Limitations\n- Extractive offline mode: verify against the cited sources."


def get_llm(backend: str = "auto", model: str = "claude-sonnet-5-5") -> LLM:
    if backend in ("auto", "anthropic"):
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
            try:
                return AnthropicLLM(model)
            except Exception:
                if backend == "anthropic":
                    raise
        elif backend == "anthropic":
            raise RuntimeError("ANTHROPIC_API_KEY not set")
    return OfflineLLM()
