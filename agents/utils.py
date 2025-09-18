from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from litellm import acompletion, completion
from pyagenity.utils import Message
from agents.logging_config import logger

if TYPE_CHECKING:
    from agents.resume_builder_state import ResumeBuilderState

# Constants
SECTION_NAMES = [
    "skills", "experiences", "education", "projects", "summary", "contact",
    "certificates", "publications", "languages", "recommendations", "custom",
]
MAX_MESSAGES = 30  # Hard cap on stored messages passed to the graph each turn
RECURSION_LIMIT = 50  # Safety cap for graph internal step traversals
ALIGNMENT_TARGET = 90

SECTION_ROUTE_PATTERNS = {
    "skills": ["/skills", "go to skills", "switch to skills"],
    "experiences": ["/experiences", "go to experiences", "switch to experiences", "experience section"],
    "education": ["/education", "go to education", "education section"],
    "projects": ["/projects", "go to projects", "project section"],
    "summary": ["/summary", "go to summary", "summary section"],
    "contact": ["/contact", "go to contact", "contact section"],
    "certificates": ["/certificates", "go to certificates", "certificates section"],
    "publications": ["/publications", "go to publications", "publications section"],
    "languages": ["/languages", "go to languages", "languages section"],
    "recommendations": ["/recommendations", "go to recommendations", "recommendations section"],
    "custom": ["/custom", "go to custom", "custom section"],
}

# LLM Configuration
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OFFLINE_MODE = GOOGLE_API_KEY is None

def safe_extract_text(resp: Any) -> Optional[str]:
    """Extract the assistant text from common litellm ModelResponse shapes."""
    try:
        if hasattr(resp, "choices") and resp.choices:
            c0 = resp.choices[0]
            if hasattr(c0, "message") and c0.message:
                return getattr(c0.message, "content", None)
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if isinstance(resp, dict):
            if "choices" in resp and resp["choices"]:
                ch = resp["choices"][0]
                if isinstance(ch, dict):
                    if "message" in ch and isinstance(ch["message"], dict):
                        return ch["message"].get("content")
            if "candidates" in resp and resp["candidates"]:
                return resp["candidates"][0].get("content")
            if "content" in resp:
                return resp.get("content")
    except Exception as e:
        logger.warning(f"Error extracting text from response: {e}")
    return None

def maybe_print_usage(resp: Any, label: str = ""):
    """Best-effort token usage printer for common shapes returned by litellm."""
    try:
        if hasattr(resp, "usage") and resp.usage:
            u = resp.usage
            prompt = getattr(u, "prompt_tokens", getattr(u, "prompt_token_count", None))
            completion_t = getattr(u, "completion_tokens", getattr(u, "candidates_token_count", None))
            total = getattr(u, "total_tokens", getattr(u, "total_token_count", None))
            print(f"[Token Usage {label}] prompt={prompt} completion={completion_t} total={total}")
        elif isinstance(resp, dict) and "usage" in resp:
            u = resp["usage"]
            print(f"[Token Usage {label}] prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} total={u.get('total_tokens')}")
    except Exception as e:
        logger.warning(f"Error printing token usage: {e}")

def extract_and_validate_json(raw_text: str) -> Dict[str, Any]:
    """
    Extract JSON from raw text and validate basic structure.
    Raises ValueError if no valid JSON found or required keys missing.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response from LLM")
    
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
    if not json_match:
        return {"action": "answer", "route": None, "answer": raw_text.strip()}
    
    json_text = json_match.group(0)
    try:
        parsed = json.loads(json_text)
        
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")
        
        if "action" not in parsed:
            raise ValueError("Missing required 'action' key in JSON response")
        
        valid_actions = {"answer", "route", "stay", "switch", "exit", "apply"}
        if parsed["action"] not in valid_actions:
            logger.warning(f"Unexpected action '{parsed['action']}', treating as 'answer'")
            parsed["action"] = "answer"
        
        return parsed
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

async def call_llm_json_decision(system_prompt: str, user_payload: Dict[str, Any], state: "Optional[ResumeBuilderState]" = None) -> Dict[str, Any]:
    """
    Call the LLM (async) and expect JSON decision output with robust error handling.
    """
    user_text = json.dumps(user_payload, separators=(",", ":"), ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    extra = {}
    if GOOGLE_API_KEY:
        extra["api_key"] = GOOGLE_API_KEY

    if OFFLINE_MODE:
        print("Offline mode: Using deterministic fallback for LLM decision.")
        return {"action": "answer", "route": None, "answer": "(Offline) I received your query and will help once an API key is configured."}
    
    try:
        resp = await acompletion(model=LLM_MODEL, messages=messages, temperature=0.5, **extra)
        print(f"LLM raw decision response: {resp}")
        maybe_print_usage(resp, "router")
        raw = safe_extract_text(resp) or ""
        
        return extract_and_validate_json(raw)
        
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        return {"action": "answer", "route": None, "answer": f"I encountered an error processing your request. Please try again."}

def last_assistant_message(messages: List[Message]) -> Optional[Message]:
    """Find the last assistant message in a list of messages."""
    for m in reversed(messages):
        if getattr(m, "role", None) == "assistant":
            return m
    return None

def extract_assistant_text(result: Dict[str, Any]) -> str:
    """Extract assistant text from a graph result, handling various structures."""
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    a = last_assistant_message(msgs)
    if not a:
        st = result.get("state") if isinstance(result, dict) else None
        if st is not None:
            ctx = getattr(st, "context", None)
            if ctx:
                a = last_assistant_message(ctx)
    return getattr(a, "content", "(No response)")

def clone_state(old_state: "ResumeBuilderState") -> "ResumeBuilderState":
    """Create a fresh state object carrying forward key fields while resetting execution meta."""
    from agents.resume_builder_state import ResumeBuilderState
    new_state = ResumeBuilderState()
    new_state.jd_summary = old_state.jd_summary
    try:
        new_state.resume_sections = getattr(old_state, "resume_sections", None)
    except Exception:
        logger.warning("Error cloning resume_sections, setting to None.")
        new_state.resume_sections = None
    new_state.section_objects = dict(getattr(old_state, "section_objects", {}) or {})
    new_state.context = list(getattr(old_state, "context", []) or [])
    new_state.current_section = getattr(old_state, "current_section", None)
    new_state.cv_summary = dict(getattr(old_state, "cv_summary", {}) or {})
    new_state.recommended_answers = dict(getattr(old_state, "recommended_answers", {}) or {})
    return new_state

def format_ai_with_section(ai_text: str, state_obj: "Optional[ResumeBuilderState]") -> str:
    """Prefix assistant message with the current section from state."""
    sec = getattr(state_obj, "current_section", None) if state_obj else None
    sec_display = sec if sec else "none"
    return f"Current section: {sec_display}\n\n{ai_text}"
