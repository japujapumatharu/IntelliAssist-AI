PROVIDERS = ["None (Offline)", "Gemini", "OpenAI"]


def get_llm(provider, api_key=None):
    if not api_key or provider.startswith("None"):
        return None
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2)
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.2)
    return None


def ask(llm, prompt):
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content).strip()
