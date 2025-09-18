"""
resume_builder_state_and_router.py

Provides:
- ResumeBuilderState (extends AgentState; stores jd_summary, section_objects, current_section)
- async general_chat_and_section_routing node that uses the LLM to:
    * answer resume/JD queries using the given data, or
    * route to a section (by LLM decision) when the user asks or LLM infers routing is best,
    * if already in a section, immediately route to that section (no interference).
"""

import json
from typing import Any, Dict

from litellm import completion

from agents.resume_builder_state import ResumeBuilderState
from agents.utils import (
    logger,
    LLM_MODEL,
    GOOGLE_API_KEY,
    OFFLINE_MODE,
    safe_extract_text,
    maybe_print_usage,
    call_llm_json_decision,
    SECTION_NAMES,
)
from agents.prompts import GENERAL_CHAT_ROUTING_PROMPT


async def general_chat_and_section_routing(state: ResumeBuilderState, config: Dict[str, Any]) -> Any:
    """
    Master node behavior (non-hardcoded):
      1. If state.current_section is set -> return {"route": current_section} immediately.
      2. Otherwise, send jd_summary + compact section_objects + user query to LLM with a tiny system prompt
         requesting JSON output indicating either:
            - action = "answer" (assistant should answer directly, don't route)
            - action = "route" (assistant suggests routing to <section>)
         Also return updated per-section metadata for each relevant section.
      3. Node updates state.section_objects and either appends assistant answer (action=answer)
         or returns {"route": "<section>"} (action=route).
    """

    print("\n--- Entering General Chat Section ---")
    last_msg = state.context[-1] if state.context else None
    first_turn = not state.context or (last_msg and getattr(last_msg, "role", "") != "user")
    user_text = ""
    if not first_turn:
        user_text = getattr(last_msg, "content", "") or ""
    else:
        user_text = "INITIAL_GREETING: Greet the user, summarize JD, list sections with alignment and missing requirements, invite next action."

    if state.current_section:
        route_json = {"route": state.current_section}
        state.context.append(state.make_message("assistant", json.dumps(route_json)))
        return state
    
    compact_sections = {}
    for s in SECTION_NAMES: # Iterate over all possible section names
        so = state.section_objects.get(s, {})
        compact_sections[s] = {
            "section_name": s,
            "alignment_score": so.get("alignment_score"),
            "missing_requirements": so.get("missing_requirements", []),
        }

    ai_messages = []
    human_messages = []
    
    for msg in reversed(state.context):
        if msg.role == 'assistant' and len(ai_messages) < 5:
            ai_messages.insert(0, msg.content)
        elif msg.role == 'user' and len(human_messages) < 5:
            human_messages.insert(0, msg.content)
        
        if len(ai_messages) >= 5 and len(human_messages) >= 5:
            break

    payload = {
        "user_query": user_text,
        "conversation_context": {
            "ai_messages": ai_messages,
            "human_messages": human_messages
        }
    }
    available_sections = ', '.join(SECTION_NAMES) # Use SECTION_NAMES directly

    system_prompt = GENERAL_CHAT_ROUTING_PROMPT.format(
        available_sections=available_sections,
        compact_sections_json=json.dumps(compact_sections, separators=(",", ":"))
    )
        
    try:
        parsed = await call_llm_json_decision(system_prompt, payload)
    except Exception as e:
        logger.error(f"Error in general_chat_and_section_routing LLM call: {e}")
        if state.current_section:
            return state
        fallback_system = "You are a concise resume assistant. Answer the user using the JD and sections provided."
        fallback_msg = f"JD: {state.jd_summary}\nSections: {json.dumps(compact_sections)}\nUser query: {user_text}"
        try:
            extra = {}
            if GOOGLE_API_KEY:
                extra["api_key"] = GOOGLE_API_KEY
            resp = completion(model=LLM_MODEL, messages=[{"role":"system","content":fallback_system},{"role":"user","content":fallback_msg}], max_completion_tokens=200, **extra)
            maybe_print_usage(resp, "fallback_answer")
            text = safe_extract_text(resp) or "Sorry — I couldn't generate a response."
        except Exception as fallback_e:
            logger.error(f"Error during fallback LLM call: {fallback_e}")
            text = "Sorry — analysis temporarily unavailable. Try again later."
        state.context.append(state.make_message("assistant", text))
        return state

    action = parsed.get("action")
    route_to = parsed.get("route")
    answer_text = parsed.get("answer")

    if action == "route" and route_to and route_to in SECTION_NAMES: # Use SECTION_NAMES
        state.current_section = route_to
        confirm_json = {"route": route_to, "note": parsed.get("reason") or f"Routing to '{route_to}'"}
        state.context.append(state.make_message("assistant", json.dumps(confirm_json)))
        return state

    if action == "answer":
        if not answer_text:
            answer_text = "I cannot determine an answer — please provide more details or ask to route to a specific section."
        state_response = state.make_message("assistant", answer_text)
        state.context.append(state_response)
        return state_response

    fallback = "I couldn't decide automatically. Would you like me to (1) analyze sections to suggest improvements, or (2) open a specific section? Use /skills, /experiences, /education, or /projects."
    state_response = state.make_message("assistant", fallback)
    state.context.append(state_response)
    return state_response