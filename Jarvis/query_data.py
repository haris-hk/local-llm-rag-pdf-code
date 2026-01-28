import argparse
import re
from datetime import datetime, timezone
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("Warning: duckduckgo-search not installed. Web search disabled. Install with: pip install duckduckgo-search")

from populate_database import prune_stale_chunks, DEFAULT_TTL_DAYS, get_db

PROMPT_TEMPLATE = """
Answer the question based only on the following context (which may include code, documentation, or text files):

{context}

---

Answer the question based on the above context. If the context contains code, you can reference it, explain it, or write similar code as needed: {question}
"""

PROMPT_TEMPLATE_THINKING = """
Answer the question based only on the following context (which may include code, documentation, or text files):

{context}

---

Think through this step-by-step, then answer the question. If the context contains code, analyze it carefully before responding: {question}
"""

PROMPT_TEMPLATE_CODE = """
You are a helpful coding assistant. Use the following code and documentation context to answer the question:

{context}

---

Based on the code context above, answer the following question. Provide code examples when relevant and explain your reasoning: {question}
"""

PROMPT_TEMPLATE_WEB = """
Answer the question using the following information gathered from the internet:

{context}

---

Based on the web search results above, provide a comprehensive answer to: {question}

If the search results don't fully answer the question, say so and provide what information is available.
"""

PROMPT_TEMPLATE_WEB_THINKING = """
Answer the question using the following information gathered from the internet:

{context}

---

Think through the search results step-by-step, then provide a comprehensive answer to: {question}

If the search results don't fully answer the question, say so and provide what information is available.
"""

PROMPT_TEMPLATE_COMBINED = """
Answer the question using the following context from your documents AND web search results:

**Your Documents:**
{doc_context}

**Web Search Results:**
{web_context}

---

Based on both your documents and the web search results, provide a comprehensive answer to: {question}
"""

# Keywords that trigger document search
DOC_KEYWORDS = [
    r'\bdocument[s]?\b', r'\bpdf[s]?\b', r'\bfile[s]?\b', r'\bupload(ed|s)?\b',
    r'\bsearch\s+(the\s+)?(docs|documents|files)\b', r'\bin\s+(the\s+)?(docs|documents|files|pdf)\b',
    r'\bfrom\s+(the\s+)?(docs|documents|files|pdf)\b', r'\baccording\s+to\b',
    r'\bwhat\s+does\s+(the|my)\b', r'\blook\s+(up|into|at)\b', r'\bfind\s+in\b',
    r'\breference\b', r'\bcite\b', r'\bsource[s]?\b',
    # Code-related keywords
    r'\bcode(base)?\b', r'\bfunction\b', r'\bclass(es)?\b', r'\bmethod\b',
    r'\bimport\b', r'\bmodule\b', r'\bscript\b', r'\bimplementation\b',
    r'\b(my|the|this)\s+(project|repo|repository|codebase)\b',
    r'\bhow\s+(does|is)\s+(the|my|this)\b', r'\bwhere\s+(is|are|does)\b',
    r'\bshow\s+me\b', r'\bexplain\s+(the|this|my)\b',
    r'\bvariable\b', r'\bapi\b', r'\bendpoint\b', r'\broute\b',
    r'\.(py|js|ts|java|cpp|c|html|css)\b'
]

# Keywords that suggest complex reasoning is needed
THINKING_KEYWORDS = [
    r'\bexplain\b', r'\bwhy\b', r'\bhow\s+does\b', r'\bcompare\b', r'\banalyze\b',
    r'\bbreak\s*down\b', r'\bstep[\s-]*by[\s-]*step\b', r'\breason(ing)?\b',
    r'\bthink\s*(through|about)?\b', r'\bsolve\b', r'\bcalculate\b', r'\bderive\b',
    r'\bprove\b', r'\bdebug\b', r'\bwhat\s+if\b', r'\btrade[\s-]*off[s]?\b',
    r'\bpros?\s*(and|&)\s*cons?\b', r'\badvantages?\b', r'\bdisadvantages?\b',
    # Code-related thinking triggers
    r'\brefactor\b', r'\boptimize\b', r'\bimprove\b', r'\bbug\b', r'\berror\b',
    r'\bfix\b', r'\bissue\b', r'\bproblem\b', r'\barchitecture\b', r'\bdesign\b',
    r'\bwrite\s+(a|the|some)?\s*(code|function|class|script)\b',
    r'\bcreate\s+(a|the)?\s*(function|class|component|module)\b',
    r'\bimplement\b', r'\badd\s+(a|the)?\s*(feature|function|method)\b'
]

# Keywords that trigger web search
WEB_SEARCH_KEYWORDS = [
    r'\bsearch\s+(the\s+)?(web|internet|online|google)\b',
    r'\blook\s+up\s+(online|on\s+the\s+internet)\b',
    r'\bwhat\s+is\s+(the\s+)?(latest|current|recent|new(est)?)\b',
    r'\bnews\s+(about|on|regarding)\b',
    r'\b(latest|current|recent|today\'?s?)\s+(news|updates?|information)\b',
    r'\bwho\s+(is|was|are|were)\b',
    r'\bwhen\s+(did|was|is|will)\b',
    r'\bwhere\s+(is|are|can\s+i)\b',
    r'\bhow\s+to\b', r'\btutorial\b', r'\bguide\b',
    r'\b(weather|stock|price|score)\b',
    r'\b20[0-9]{2}\b',  # Years like 2024, 2025
    r'\btoday\b', r'\byesterday\b', r'\bthis\s+(week|month|year)\b',
    r'\bonline\b', r'\binternet\b', r'\bwebsite\b',
    r'\btrending\b', r'\bpopular\b', r'\bviral\b',
    r'\breview[s]?\b', r'\brating[s]?\b',
    r'\bdownload\b', r'\binstall\b',
    r'\bdocumentation\b', r'\bofficial\b',
]


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo and return results.
    """
    if not WEB_SEARCH_AVAILABLE:
        return []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []


def format_web_results(results: list[dict]) -> str:
    """Format web search results into a context string."""
    if not results:
        return "No web results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        body = result.get('body', 'No description')
        href = result.get('href', '')
        formatted.append(f"**[{i}] {title}**\nURL: {href}\n{body}")
    
    return "\n\n".join(formatted)


def should_use_web_search(query_text: str, web_search: str = "auto") -> bool:
    """
    Determine if web search should be used.
    
    web_search can be:
    - "on": Always search the web
    - "off": Never search the web
    - "auto": Search when query seems to need current/external info
    """
    if web_search == "on":
        return True
    if web_search == "off":
        return False
    
    # Auto mode: check for keywords suggesting web search is needed
    query_lower = query_text.lower()
    for pattern in WEB_SEARCH_KEYWORDS:
        if re.search(pattern, query_lower):
            return True
    
    return False


def should_use_rag(query_text: str, use_docs: str = "auto") -> bool:
    """
    Determine if RAG should be used based on mode and query content.
    """
    if use_docs == "always":
        return True
    if use_docs == "never":
        return False
    
    query_lower = query_text.lower()
    for pattern in DOC_KEYWORDS:
        if re.search(pattern, query_lower):
            return True
    return False


def should_use_thinking(query_text: str, thinking_mode: str = "auto") -> bool:
    """
    Determine if thinking/reasoning mode should be enabled.
    
    thinking_mode can be:
    - "on": Always enable thinking
    - "off": Never enable thinking (faster responses)
    - "auto": Enable for complex queries that need reasoning
    """
    if thinking_mode == "on":
        return True
    if thinking_mode == "off":
        return False
    
    # Auto mode: check for keywords suggesting complex reasoning
    query_lower = query_text.lower()
    for pattern in THINKING_KEYWORDS:
        if re.search(pattern, query_lower):
            return True
    
    # Also enable for longer queries (likely more complex)
    if len(query_text.split()) > 20:
        return True
    
    return False


def clean_thinking_response(response: str, thinking_enabled: bool) -> str:
    """
    Clean up the response from thinking mode.
    Qwen3 outputs thinking in <think>...</think> tags.
    """
    # Check if response has thinking tags (regardless of mode, model might still think)
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, response, re.DOTALL)
    
    if match:
        thinking_content = match.group(1).strip()
        # Remove the thinking tags from response
        final_response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
        
        # If thinking was enabled, show the reasoning process
        if thinking_enabled and thinking_content:
            # Format with clear visual separation
            formatted_thinking = thinking_content.replace('\n', '\n> ')
            return f"💭 **Internal Reasoning:**\n> {formatted_thinking}\n\n---\n\n**Answer:**\n{final_response}"
        else:
            # Just return the final response without thinking
            return final_response
    
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--model",
        type=str,
        default="jarvis",
        help="Ollama model name to query (default: jarvis)",
    )
    parser.add_argument(
        "--use-docs",
        type=str,
        default="auto",
        choices=["always", "never", "auto"],
        help="Document search mode: always, never, or auto (default: auto)",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        default="auto",
        choices=["on", "off", "auto"],
        help="Thinking/reasoning mode: on, off, or auto (default: auto)",
    )
    parser.add_argument(
        "--web",
        type=str,
        default="auto",
        choices=["on", "off", "auto"],
        help="Web search mode: on, off, or auto (default: auto)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="global",
        help="Session identifier used to scope document retrieval",
    )
    args = parser.parse_args()
    query_text = args.query_text
    result = query_rag(
        query_text, 
        model_name=args.model, 
        use_docs=args.use_docs,
        thinking_mode=args.thinking,
        web_search=args.web,
        session_id=args.session_id,
    )
    print(f"Response: {result['response']}")
    print(f"Sources: {result['sources']}")
    print(f"Used RAG: {result['used_rag']}")
    print(f"Used Thinking: {result['used_thinking']}")
    print(f"Used Web Search: {result['used_web_search']}")


def query_rag(
    query_text: str, 
    model_name: str = "jarvis", 
    use_docs: str = "auto",
    thinking_mode: str = "auto",
    web_search: str = "auto",
    session_id: str | None = "global",
):
    """
    Run a query and return response + sources.
    
    use_docs: "always" | "never" | "auto"
    thinking_mode: "on" | "off" | "auto"
    web_search: "on" | "off" | "auto"
    """
    use_thinking = should_use_thinking(query_text, thinking_mode)
    use_web = should_use_web_search(query_text, web_search)
    target_session = session_id or "global"
    
    # For Qwen3 models, prepend /think or /no_think to enable/disable thinking
    if use_thinking:
        thinking_prefix = "/think "
    else:
        thinking_prefix = "/no_think "
    
    model = OllamaLLM(model=model_name)
    
    # Determine if we should use RAG
    use_rag = should_use_rag(query_text, use_docs)
    
    # Perform web search if needed
    web_context = ""
    web_sources = []
    if use_web and WEB_SEARCH_AVAILABLE:
        print("🌐 Searching the web...")
        web_results = search_web(query_text, max_results=5)
        if web_results:
            web_context = format_web_results(web_results)
            web_sources = [r.get('href', '') for r in web_results]
    
    # Case 1: No RAG, No Web - Pure LLM
    if not use_rag and not web_context:
        query_with_thinking = thinking_prefix + query_text
        response_text = model.invoke(query_with_thinking)
        response_text = clean_thinking_response(response_text, use_thinking)
        
        return {
            "response": response_text,
            "sources": [],
            "web_sources": [],
            "context": "",
            "used_rag": False,
            "used_thinking": use_thinking,
            "used_web_search": False,
        }
    
    # Case 2: Web only, no RAG
    if not use_rag and web_context:
        if use_thinking:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WEB_THINKING)
        else:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WEB)
        
        prompt = prompt_template.format(context=web_context, question=query_text)
        prompt_with_thinking = thinking_prefix + prompt
        
        response_text = model.invoke(prompt_with_thinking)
        response_text = clean_thinking_response(response_text, use_thinking)
        
        return {
            "response": response_text,
            "sources": [],
            "web_sources": web_sources,
            "context": web_context,
            "used_rag": False,
            "used_thinking": use_thinking,
            "used_web_search": True,
        }
    
    # Case 3 & 4: RAG mode (with or without web)
    db = get_db()
    prune_stale_chunks(db, ttl_days=DEFAULT_TTL_DAYS)

    results = db.similarity_search_with_score(
        query_text,
        k=5,
        filter={"session_id": target_session},
    )

    now = datetime.now(timezone.utc).isoformat()
    hit_ids = [doc.metadata.get("id") for doc, _score in results if doc.metadata.get("id")]
    if hit_ids:
        db._collection.update(ids=hit_ids, metadatas=[{"last_accessed": now}] * len(hit_ids))

    doc_context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Case 3: RAG + Web combined
    if web_context:
        combined_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_COMBINED)
        prompt = combined_template.format(
            doc_context=doc_context, 
            web_context=web_context, 
            question=query_text
        )
    else:
        # Case 4: RAG only
        if use_thinking:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_THINKING)
        else:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=doc_context, question=query_text)
    
    prompt_with_thinking = thinking_prefix + prompt

    response_text = model.invoke(prompt_with_thinking)
    response_text = clean_thinking_response(response_text, use_thinking)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return {
        "response": response_text,
        "sources": sources,
        "web_sources": web_sources,
        "context": doc_context,
        "used_rag": True,
        "used_thinking": use_thinking,
        "used_web_search": bool(web_context),
    }


if __name__ == "__main__":
    main()
