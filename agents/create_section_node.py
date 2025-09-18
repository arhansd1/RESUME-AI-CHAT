import json
import re
from typing import Any, Dict, List, Callable, Optional, Tuple

from agents.resume_builder_state import ResumeBuilderState
from agents.utils import (
    logger,
    SECTION_NAMES,
    SECTION_ROUTE_PATTERNS,
    LLM_MODEL,
    GOOGLE_API_KEY,
    OFFLINE_MODE,
    safe_extract_text,
    maybe_print_usage,
    extract_and_validate_json,
    call_llm_json_decision,
)
from agents.prompts import (
    CV_SUMMARY_ANALYSIS_PROMPT,
    SECTION_IMPROVEMENT_PROMPT,
    SECTION_NODE_SYSTEM_PROMPT,
)
from litellm import acompletion


async def update_cv_summary(
    state: ResumeBuilderState, 
    section_name: str, 
    recommended_questions: List[str], 
    recommended_answers: List[str]
) -> Dict[str, str]:
    """
    Update cv_summary based on new answers from a section.
    Format: {"section_name": "+positive_point\n+positive_point\n-negative_point", ...}
    """
    print(f"🔄 UPDATING CV Summary for {section_name}...")
    
    current_cv_summary = state.cv_summary.copy() if hasattr(state, 'cv_summary') else {}
    
    qa_pairs = []
    for i, (question, answer) in enumerate(zip(recommended_questions, recommended_answers)):
        if answer and answer.strip():
            qa_pairs.append({
                "question": question,
                "answer": answer.strip(),
                "index": i
            })
    
    if not qa_pairs:
        print("No new answers to process for CV summary")
        return current_cv_summary

    summary_prompt = CV_SUMMARY_ANALYSIS_PROMPT.format(
        section_name=section_name,
        current_cv_summary=json.dumps(current_cv_summary),
        qa_pairs=json.dumps(qa_pairs)
    )
    
    try:
        messages = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": json.dumps({
                "section": section_name,
                "current_summary": current_cv_summary,
                "qa_pairs": qa_pairs
            }, separators=(",", ":"))}
        ]
        
        extra = {}
        if GOOGLE_API_KEY:
            extra["api_key"] = GOOGLE_API_KEY

        if OFFLINE_MODE:
            print("Offline mode: Using enhanced basic CV summary update")
            section_points = []
            
            for qa in qa_pairs:
                answer = qa["answer"].lower()
                
                if any(neg in answer for neg in [
                    "no ", "don't", "haven't", "never", "not ", "lack", 
                    "no experience", "not familiar", "dont have", "do not have",
                    "unfamiliar", "missing", "limited experience"
                ]):
                    section_points.append(f"-{qa['answer'][:100]}")
                else:
                    section_points.append(f"+{qa['answer'][:100]}")
            
            current_cv_summary[section_name] = "\n".join(section_points)
            return current_cv_summary
        
        resp = await acompletion(model=LLM_MODEL, messages=messages, **extra)
        maybe_print_usage(resp, f"cv_summary_{section_name}")
        raw = safe_extract_text(resp) or ""
        
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                result = json.loads(json_text)
                
                if "section_points" in result:
                    section_points = result["section_points"]
                    current_cv_summary[section_name] = section_points
                    
                    print(f"✅ Updated {section_name} in CV summary")
                    return current_cv_summary
                    
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in CV summary: {e}")
        
        section_points = []
        for qa in qa_pairs:
            answer = qa["answer"].lower()
            if any(neg_phrase in answer for neg_phrase in [
                "no ", "don't", "haven't", "never", "not ", "lack", 
                "no experience", "not familiar", "unfamiliar"
            ]):
                section_points.append(f"-{qa['answer'][:100]}")
            else:
                section_points.append(f"+{qa['answer'][:100]}")
        
        current_cv_summary[section_name] = "\n".join(section_points)
        return current_cv_summary
        
    except Exception as e:
        logger.error(f"❌ Error in update_cv_summary: {e}")
        return current_cv_summary

async def apply_section_changes(state: ResumeBuilderState, section_name: str, updated_content: str) -> Dict[str, Any]:
    """
    Apply changes to a section and re-analyze it against the job description.
    Updates alignment score, missing requirements, recommended questions, and CV summary.
    
    Args:
        state: Current resume builder state
        section_name: Name of the section to update
        updated_content: New content for the section
    
    Returns:
        Dictionary with new analysis results
    """
    print(f"🔄 APPLYING changes to {section_name}...")
    
    if not hasattr(state, 'resume_sections'):
        state.resume_sections = {}
    state.resume_sections[section_name] = updated_content
    
    section_questions = state.section_objects.get(section_name, {}).get("recommended_questions", [])
    section_answers = state.recommended_answers.get(section_name, [])
    
    if section_questions and section_answers:
        try:
            updated_cv_summary = await update_cv_summary(
                state, section_name, section_questions, section_answers
            )
            state.cv_summary = updated_cv_summary
            print(f"📊 CV Summary now has {len(state.cv_summary)} points")
        except Exception as e:
            logger.warning(f"⚠️ Error updating CV summary: {e}")
            
    state.recommended_answers[section_name] = []
    
    analysis_payload = {
        "jd_summary": state.jd_summary or "",
        "section_name": section_name,
        "section_content": updated_content,
        "original_section_data": state.section_objects.get(section_name, {}),
        "cv_summary": getattr(state, 'cv_summary', {})
    }
    
    analysis_prompt = SECTION_IMPROVEMENT_PROMPT.format(
        jd_summary=analysis_payload["jd_summary"],
        section_name=analysis_payload["section_name"],
        section_content=analysis_payload["section_content"],
        cv_summary=json.dumps(analysis_payload["cv_summary"], indent=2),
    )
    
    try:
        messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": json.dumps(analysis_payload, separators=(",", ":"))}
        ]
        
        extra = {}
        if GOOGLE_API_KEY:
            extra["api_key"] = GOOGLE_API_KEY

        if OFFLINE_MODE:
            return {
                "alignment_score": 75,
                "missing_requirements": ["More specific examples needed"],
                "recommended_questions": [f"Can you add more specific examples to your {section_name}?"],
                "analysis_summary": "Offline mode - please configure API key for full analysis"
            }
        
        resp = await acompletion(model=LLM_MODEL, messages=messages, **extra)
        print(f"LLM raw analysis response: {resp}")
        maybe_print_usage(resp, f"apply_{section_name}")
        raw = safe_extract_text(resp) or ""
        
        analysis_result = {}
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                analysis_result = json.loads(json_text)
            else:
                logger.warning(f"No JSON found in analysis response: {raw}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in analysis: {e}")
            logger.warning(f"Raw response: {raw}")
        
        analysis_result.setdefault("alignment_score", 70)
        analysis_result.setdefault("missing_requirements", [])
        analysis_result.setdefault("recommended_questions", [])
        analysis_result.setdefault("analysis_summary", "Section updated successfully")
        
        if section_name not in state.section_objects:
            state.section_objects[section_name] = {}
        
        state.section_objects[section_name].update({
            "alignment_score": analysis_result["alignment_score"],
            "missing_requirements": analysis_result["missing_requirements"],
            "recommended_questions": analysis_result["recommended_questions"],
            "last_updated": "just_now"
        })
        
        if analysis_result["recommended_questions"]:
            state.recommended_answers[section_name] = [""] * len(analysis_result["recommended_questions"])
        else:
            state.recommended_answers[section_name] = []
        
        print(f"✅ {section_name} updated - New alignment score: {analysis_result['alignment_score']}%")
        print(f"📋 New questions: {len(analysis_result['recommended_questions'])}")
        print(f"📊 CV Summary points: {len(getattr(state, 'cv_summary', {}))}")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ Error in apply_section_changes: {e}")
        return {
            "alignment_score": 70,
            "missing_requirements": ["Analysis error - please try again"],
            "recommended_questions": [f"Let's continue improving your {section_name}. What specific details can you add?"],
            "analysis_summary": f"There was an error analyzing the updated {section_name}. The changes have been saved."
        }

def detect_question_matches(user_answer: str, questions: List[str]) -> List[Tuple[int, str, float]]:
    """
    Detect which questions the user's answer might be responding to using keyword matching.
    Returns list of (question_index, question_text, confidence_score) sorted by confidence.
    """
    matches = []
    user_words = set(user_answer.lower().split())
    
    for idx, question in enumerate(questions):
        question_words = set(question.lower().split())
        
        common_words = user_words & question_words
        if len(common_words) > 0:
            confidence = len(common_words) / max(len(question_words), 1)
            matches.append((idx, question, confidence))
    
    return sorted(matches, key=lambda x: x[2], reverse=True)


def detect_direct_routing(text: str) -> Optional[str]:
    t = text.lower()
    for sec, pats in SECTION_ROUTE_PATTERNS.items():
        if any(p in t for p in pats):
            return sec
    return None

def safe_initialize_answers(state: ResumeBuilderState, section_name: str, questions: List[str]) -> None:
    """Safely initialize answers array for a section with proper synchronization."""
    if section_name not in state.recommended_answers:
        state.recommended_answers[section_name] = [""] * len(questions)
    else:
        current_answers = state.recommended_answers[section_name]
        if len(current_answers) != len(questions):
            new_answers = [""] * len(questions)
            for i in range(min(len(current_answers), len(questions))):
                new_answers[i] = current_answers[i]
            state.recommended_answers[section_name] = new_answers


def create_section_node(section: str) -> Callable:
    """Create an async node implementing improvement workflow for a section."""
    async def _node(state: ResumeBuilderState, config: Dict[str, Any]) -> Any:
        print(f"--- Entering {section.capitalize()} Section ---")
        state.src_node = state.current_section
        user_msg = next((m for m in reversed(state.context) if getattr(m, "role", "") == "user"), None)
        user_text = getattr(user_msg, "content", "") if user_msg else ""

        if user_text:
            direct = detect_direct_routing(user_text)
            if direct and direct != section:
                state.current_section = direct
                state.context.append(state.make_message("assistant", json.dumps({"route": direct, "note": "Switching section"})))
                return state
            if any(kw in user_text.lower() for kw in ["/exit", "back to chat", "general chat", "leave section"]):
                state.current_section = None
                state.context.append(state.make_message("assistant", "Leaving section. Back to general analysis."))
                return state

        compact_sections = {}
        for s, data in state.section_objects.items():
            compact_sections[s] = {
                "section_name": s,
                "alignment_score": data.get("alignment_score"),
                "missing_requirements": data.get("missing_requirements", []),
                "recommended_questions": data.get("recommended_questions", []),
            }

        recommended_questions = []
        current_answers = []
        if state.current_section and state.current_section in compact_sections:
            section_data = compact_sections[state.current_section]
            recommended_questions = section_data.get("recommended_questions", [])
            
            safe_initialize_answers(state, state.current_section, recommended_questions)
            current_answers = state.recommended_answers[state.current_section]

        ai_messages = []
        human_messages = []
        for msg in reversed(state.context):
            if msg.role == 'assistant' and len(ai_messages) < 5:
                ai_messages.insert(0, msg.content)
            elif msg.role == 'user' and len(human_messages) < 5:
                human_messages.insert(0, msg.content)
            
            if len(ai_messages) >= 5 and len(human_messages) >= 5:
                break

        current_section_content = ""
        if state.current_section and hasattr(state, 'resume_sections') and state.current_section in state.resume_sections:
            current_section_content = state.resume_sections[state.current_section]
            if isinstance(current_section_content, list):
                current_section_content = "\n".join(str(item) for item in current_section_content)
            elif isinstance(current_section_content, dict):
                current_section_content = "\n".join(f"{k}: {v}" for k, v in current_section_content.items())
        
        current_schema_section = {}
        if state.current_section and hasattr(state, 'resume_schema') and state.current_section in state.resume_schema:
            current_schema_section = state.resume_schema[state.current_section]
        
        current_schema_section_json = json.dumps(current_schema_section)
        
        print(f"Current Section schema: {json.dumps(current_schema_section_json, indent=2)}")

        current_section_data = compact_sections.get(state.current_section, {})
        transformation_level = 7

        payload = {
            "user_query": user_text,
            "conversation_context": {
                "ai_messages": ai_messages,
                "human_messages": human_messages
            }
        }
        
        system_prompt = SECTION_NODE_SYSTEM_PROMPT.format(
            current_section=state.current_section,
            current_section_data=json.dumps(current_section_data, indent=2),
            current_section_content=current_section_content,
            current_schema_section_json=current_schema_section_json,
            current_answers=current_answers,
            transformation_level=transformation_level,
        )

        try:
            parsed = await call_llm_json_decision(system_prompt, payload, state=state)
            
            print(f"LLM Response: {json.dumps(parsed, indent=2)}")
            
            if parsed.get("action") == "apply" and state.current_section:
                updated_content = parsed.get("updated_section_content", "")
                if updated_content:
                    analysis_result = await apply_section_changes(state, state.current_section, updated_content)
                    
                    confirmation_msg = (
                        f"✅ Applied changes to {state.current_section}!\n\n"
                        f"📊 New alignment score: {analysis_result['alignment_score']}%\n"
                        f"{analysis_result.get('analysis_summary', '')}\n\n"
                    )
                    
                    if analysis_result.get("recommended_questions"):
                        confirmation_msg += f"Let's continue improving this section with {len(analysis_result['recommended_questions'])} more targeted questions."
                    else:
                        confirmation_msg += "🎉 This section looks complete! You can switch to another section or continue refining."
                    
                    state.context.append(state.make_message("assistant", confirmation_msg))
                    
                    print(f"🔄 Staying in {state.current_section} with updated analysis")
                    return state
                else:
                    state.context.append(state.make_message(
                        "assistant", 
                        "I couldn't find the updated content to apply. Could you please confirm the changes you'd like to make?"
                    ))
                    return state
            
            if state.current_section and ('question_matches' in parsed or 'updated_answers' in parsed):
                if state.current_section not in state.recommended_answers:
                    state.recommended_answers[state.current_section] = [''] * len(recommended_questions)
                
                if 'question_matches' in parsed and 'updated_answers' in parsed:
                    question_matches = parsed.get('question_matches', [])
                    updated_answers = parsed.get('updated_answers', [])
                    
                    if isinstance(question_matches, list) and isinstance(updated_answers, list):
                        for match_idx in question_matches:
                            if (isinstance(match_idx, int) and 
                                0 <= match_idx < len(state.recommended_answers[state.current_section]) and
                                match_idx < len(updated_answers) and
                                updated_answers[match_idx]):
                                
                                state.recommended_answers[state.current_section][match_idx] = updated_answers[match_idx]
                                print(f"Updated answer {match_idx} to: {updated_answers[match_idx]}")
                
                elif 'updated_answers' in parsed and isinstance(parsed['updated_answers'], list):
                    updated_answers = parsed['updated_answers']
                    for i, answer in enumerate(state.recommended_answers[state.current_section]):
                        if i < len(updated_answers) and updated_answers[i] and (not answer or len(str(answer).strip()) < 5):
                            state.recommended_answers[state.current_section][i] = updated_answers[i]
                            print(f"Updated answer {i} to: {updated_answers[i]}")
                    else:
                        matches = detect_question_matches(user_text, recommended_questions)
                        if matches and updated_answers:
                            best_match_idx = matches[0][0]
                            new_answer = next((ans for ans in updated_answers if ans), None)
                            if new_answer and best_match_idx < len(state.recommended_answers[state.current_section]):
                                state.recommended_answers[state.current_section][best_match_idx] = new_answer
                                print(f"Auto-matched answer to question {best_match_idx}: {new_answer}")
                
                print("\n" + "="*80)
                print("CURRENT ANSWERS:")
                for i, (q, a) in enumerate(zip(recommended_questions, state.recommended_answers[state.current_section])):
                    print(f"{i+1}. Q: {q}\n   A: {a}\n")
                print("="*80 + "\n")     
                       
        except ValueError as e:
            print(f"JSON parsing error: {e}")
            fallback_text = "I'm having trouble understanding your request. Could you please rephrase?"
            state_response = state.make_message("assistant", fallback_text)
            state.context.append(state_response)
            return state_response
        except Exception as e:
            print(f"Unexpected error in LLM call: {e}")
            fallback_text = "I encountered an error. How can I help you with your resume?"
            state_response = state.make_message("assistant", fallback_text)
            state.context.append(state_response)
            return state_response

        state.routing_attempts = 0

        action = parsed.get("action")
        route_to = parsed.get("route")
        answer_text = parsed.get("answer", "")

        if state.current_section:
            if action == "switch" and route_to:
                if route_to in compact_sections:
                    state.current_section = route_to
                    print(f"Switching to section: {route_to}")
                    return state
                else:
                    state.context.append(state.make_message(
                        "assistant",
                        f"I couldn't find a section named '{route_to}'. "
                        f"Available sections are: {', '.join(sorted(compact_sections.keys()))}. "
                        "Please try again with one of these section names."
                    ))
                    return state
            elif action == "exit":
                state.current_section = None
                exit_msg = answer_text or "Back to general chat. How can I help you with your resume?"
                state.context.append(state.make_message("assistant", exit_msg))
                return state
            else:
                if answer_text:
                    state.context.append(state.make_message("assistant", answer_text))
                print(f"Staying in section: {state.current_section}")
                return state
        else:
            if action == "route" and route_to:
                if route_to in compact_sections:
                    state.current_section = route_to
                    print(f"Routing to section: {route_to}")
                    return state
                else:
                    state.context.append(state.make_message(
                        "assistant",
                        f"I couldn't find a section named '{route_to}'. "
                        f"Available sections are: {', '.join(sorted(compact_sections.keys()))}. "
                        "Please try again with one of these section names."
                    ))
                    return state
            else:
                if not answer_text:
                    answer_text = "I'm here to help with your resume. Would you like to work on a specific section?"
                state_response = state.make_message("assistant", answer_text)
                state.context.append(state_response)
                return state_response

    return _node