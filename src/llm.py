"""Optional LLM providers. With no key the app falls back to local extractive mode."""
from __future__ import annotations

PROVIDERS = ["None (local / extractive)", "Gemini", "OpenAI"]


def get_llm(provider: str, api_key: str | None):
    if not api_key or provider.startswith("None"):
        return None
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2)
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.2)
    return None


def ask(llm, prompt: str) -> str:
    out = llm.invoke(prompt)
    content = getattr(out, "content", out)
    if isinstance(content, list):  # some providers return content blocks
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content).strip()
