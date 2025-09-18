import gradio as gr
import httpx
from uuid import uuid4
import json

API_BASE_URL = "http://localhost:8000/v1"
MAX_MESSAGES = 30  # Mirror graph_builder cap

# --- API Helper Functions ---

async def api_get_threads():
    """List all threads from the checkpointer."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/threads")
            response.raise_for_status()
            return response.json().get("data", [])
    except Exception as e:
        print(f"Error getting threads: {e}")
        return []

def sanitize_messages(messages):
    """Keep only role/content for each message to satisfy API schema."""
    safe = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        safe.append({"role": m.get("role"), "content": m.get("content")})
    return safe

def to_state_dict(state_obj):
    """Convert initial_state pydantic object to a plain dict for JSON serialization."""
    if state_obj is None:
        return {}
    try:
        # Directly return if it's already a dictionary
        if isinstance(state_obj, dict):
            return state_obj
        
        allowed_keys = {
            "jd_summary",
            "resume_sections",
            "section_objects",
            "current_section",
            "src_node",
            "recommended_answers",
            "routing_attempts",
            "resume_schema",
            "cv_summary",
            "context",  # Keep context for message history
        }
        
        dump_obj = {}
        if hasattr(state_obj, "model_dump"):
            dump_obj = state_obj.model_dump(include=allowed_keys, exclude_none=True)
        elif hasattr(state_obj, "dict"):
            data = state_obj.dict()
            dump_obj = {k: data.get(k) for k in allowed_keys if k in data}
        
        # Sanitize context separately
        if "context" in dump_obj:
            dump_obj["context"] = sanitize_messages(dump_obj["context"])
            
        return dump_obj
    except Exception:
        return {}

async def api_invoke_graph(thread_id, messages, initial_state=None, config=None):
    """Call the /v1/graph/invoke endpoint."""
    # The API expects a list of messages with just 'role' and 'content'.
    # We'll strip out other keys that the API might have sent back in previous turns.
    messages_to_send = sanitize_messages(messages)
    payload = {
        "messages": messages_to_send,
        "initial_state": to_state_dict(initial_state) if initial_state is not None else {},
        "config": config or {"thread_id": thread_id},
        "response_granularity": "full",
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_BASE_URL}/graph/invoke", json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

async def api_put_state(thread_id, state):
    """Save state to the checkpointer."""
    if not state:
        return {"error": "No state to save."}
    try:
        async with httpx.AsyncClient() as client:
            # The state from invoke is wrapped in a 'data' key, but the PUT endpoint expects the raw state object.
            state_to_put = state.get("data", {}).get("state", {})
            if not state_to_put:
                 return {"error": "Extracted state is empty."}
            response = await client.put(f"{API_BASE_URL}/threads/{thread_id}/state", json=state_to_put)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

async def api_get_state(thread_id):
    """Get state from the checkpointer."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/threads/{thread_id}/state")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

# --- Gradio UI Helper Functions ---

def last_assistant_message(data):
    """Extract the last assistant message from API response."""
    print("API Response:", data)
    if "error" in data:
        error_message = f"(No response: {data.get('error', 'unknown reason')})"
        print("Error:", error_message)
        return error_message
    messages = data.get("data", {}).get("messages") or []
    if messages:
        for m in reversed(messages):
            if m.get("role") == "assistant":
                print("Assistant message:", m.get("content", "(Empty)"))
                return m.get("content", "(Empty)")
    # Fallback: look into state.context if messages are empty
    context = data.get("data", {}).get("state", {}).get("context", [])
    for m in reversed(context):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role == "assistant":
            print("Assistant message (from context):", content or "(Empty)")
            return content or "(Empty)"
    print("Error: No assistant message found")
    return "(No assistant message found)"

def extract_messages_or_context(api_data):
    """Return list of {role, content} from data.messages or fallback to state.context."""
    msgs = api_data.get("data", {}).get("messages") or []
    if msgs:
        return sanitize_messages(msgs)
    ctx = api_data.get("data", {}).get("state", {}).get("context", [])
    out = []
    for m in ctx:
        if isinstance(m, dict):
            out.append({"role": m.get("role"), "content": m.get("content")})
        else:
            r = getattr(m, "role", None)
            c = getattr(m, "content", None)
            if r or c:
                out.append({"role": r, "content": c})
    return out

def extract_state_info(state_data):
    """Extract displayable info from a state object."""
    if not state_data or "error" in state_data:
        return {}, {}, {}, None
    state = state_data.get("data", {}).get("state", state_data.get("data", {})) # Handle both invoke and get_state responses
    return (
        state.get("cv_summary", {}),
        state.get("resume_sections", {}),
        state.get("section_objects", {}),
        state.get("current_section", None),
    )

def format_ai_with_section(ai_text: str, current_section: str | None) -> str:
    """Prefix assistant message with the current section like normal Gradio UI."""
    sec_display = current_section if current_section else "none"
    return f"Current section: {sec_display}\n\n{ai_text}"

def extract_state_for_next_call(state_container: dict | None) -> dict:
    """Given st_latest_state_obj (invoke response or get_state result), return the inner state dict.

    - If it's an invoke response: expects shape {"data": {"state": {...}}}
    - If it's a get_state result: expects shape {"data": {...}}
    - Otherwise returns {}
    """
    if not isinstance(state_container, dict):
        return {}
    data = state_container.get("data", {})
    if isinstance(data, dict) and "state" in data and isinstance(data["state"], dict):
        return data["state"]
    if isinstance(data, dict):
        return data
    return {}

def build_api_gradio_ui(initial_state=None):
    """Builds the Gradio UI for interacting with the resume builder API."""

    with gr.Blocks(title="Resume Builder API Client") as demo:
        gr.Markdown("## Resume Builder API Client")
        gr.Markdown("Interact with the resume builder graph via its FastAPI endpoints.")

        # Hidden state management
        st_thread_id = gr.State(str(uuid4()))
        st_chat_history_api = gr.State([])
        st_latest_state_obj = gr.State({})

        with gr.Row():
            with gr.Column(scale=2):
                chat_display = gr.Chatbot(label="AI Chat", height=540, show_copy_button=True, type="tuples")
                with gr.Row():
                    user_box = gr.Textbox(placeholder="Send a message...", lines=3, show_label=False)
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear & New Thread")

            with gr.Column(scale=1):
                with gr.Accordion("Thread Management", open=True):
                    thread_id_display = gr.Textbox(label="Current Thread ID", interactive=False)
                    save_state_btn = gr.Button("Save Current State")
                    status_display = gr.Markdown("")
                with gr.Accordion("Load Thread", open=False):
                    saved_threads_dropdown = gr.Dropdown(label="Saved Threads", interactive=True)
                    refresh_threads_btn = gr.Button("Refresh List")
                    load_thread_btn = gr.Button("Load Selected Thread")
                
                gr.Markdown("### Live State from API")
                cv_summary_display = gr.JSON(label="CV Summary")
                sections_json = gr.JSON(label="Resume Sections")
                section_objs_json = gr.JSON(label="Section Objects")

        # --- Event Handlers ---

        async def on_load(thread_id):
            """Initial greeting on UI load."""
            # Mirror normal Gradio: assistant-first turn triggered by a startup message
            # API disallows system role, so use user role with SESSION_START content.
            initial_message = {"role": "user", "content": "SESSION_START"}
            api_response = await api_invoke_graph(thread_id, [initial_message], initial_state=initial_state)
            
            assistant_response = last_assistant_message(api_response)
            cv_summary, resume_sections, section_objects, current_section = extract_state_info(api_response)
            formatted_ai = format_ai_with_section(assistant_response, current_section)
            print(f"Init -> Current section: {current_section}")
            chat_history_display = [("", formatted_ai)]
            chat_history_api = extract_messages_or_context(api_response)
            
            threads = await api_get_threads()
            thread_ids = [t.get("thread_id") for t in threads if isinstance(t, dict)]

            return thread_id, chat_history_display, chat_history_api, api_response, cv_summary, resume_sections, section_objects, gr.update(choices=thread_ids)

        async def on_message(user_text, chat_history_display, chat_history_api, thread_id, latest_state_obj):
            """Handle user message submission."""
            print("\n--- New Message ---")
            print(f"User: {user_text}")
            if not user_text.strip():
                yield chat_history_display, chat_history_api, {}, {}, {}, {}, ""
                return

            chat_history_display.append((user_text, None))
            chat_history_api.append({"role": "user", "content": user_text})
            # Trim messages like normal UI to avoid unbounded growth
            if len(chat_history_api) > MAX_MESSAGES:
                chat_history_api = chat_history_api[-MAX_MESSAGES:]
                print(f"(Trimmed API chat history to {MAX_MESSAGES})")
            yield chat_history_display, chat_history_api, {}, {}, {}, {}, ""

            prev_state = extract_state_for_next_call(latest_state_obj)
            api_response = await api_invoke_graph(thread_id, chat_history_api, initial_state=prev_state)
            
            assistant_response = last_assistant_message(api_response)
            cv_summary, resume_sections, section_objects, current_section = extract_state_info(api_response)
            formatted_ai = format_ai_with_section(assistant_response, current_section)
            print(f"AI (section={current_section}): {assistant_response}")
            chat_history_display[-1] = (user_text, formatted_ai)
            
            chat_history_api = extract_messages_or_context(api_response)

            yield chat_history_display, chat_history_api, api_response, cv_summary, resume_sections, section_objects, ""

        def on_clear():
            """Reset the UI for a new conversation, but keep the thread ID."""
            return [], [], {}, {}, {}, {}, "Status: Chat cleared. Ready for new conversation."

        async def on_save_state(thread_id, state_obj):
            """Save the current state via API."""
            if not state_obj:
                return "Status: Error - No state has been captured yet."
            result = await api_put_state(thread_id, state_obj)
            if "error" in result:
                return f"Status: Error saving state - {result['error']}"
            return f"Status: State saved successfully for thread {thread_id}."

        async def on_refresh_threads():
            """Fetch the latest list of saved threads."""
            threads = await api_get_threads()
            thread_ids = [t.get("thread_id") for t in threads]
            return gr.update(choices=thread_ids, value=None)

        async def on_load_thread(thread_id_to_load):
            """Load a saved thread's state and continue the conversation."""
            if not thread_id_to_load:
                return thread_id_to_load, [], [], {}, {}, {}, {}, "Status: Please select a thread to load."

            state_response = await api_get_state(thread_id_to_load)
            if "error" in state_response:
                return thread_id_to_load, [], [], {}, {}, {}, {}, f"Status: Error loading state - {state_response['error']}"

            # The state is the "initial state" for the next turn
            initial_state = state_response.get("data", {})
            # We can start the conversation from here. Let's use a placeholder message.
            messages = [{"role": "user", "content": "Continue from loaded state."}]
            
            api_response = await api_invoke_graph(thread_id_to_load, messages, initial_state=initial_state)
            
            assistant_response = last_assistant_message(api_response)
            cv_summary, resume_sections, section_objects, current_section = extract_state_info(api_response)
            formatted_ai = format_ai_with_section(assistant_response, current_section)
            chat_history_display = [("Loaded state", formatted_ai)]
            chat_history_api = extract_messages_or_context(api_response)

            return thread_id_to_load, chat_history_display, chat_history_api, api_response, cv_summary, resume_sections, section_objects, f"Status: Loaded thread {thread_id_to_load}."

        # --- Event Listeners ---
        demo.load(
            on_load,
            inputs=[st_thread_id],
            outputs=[st_thread_id, chat_display, st_chat_history_api, st_latest_state_obj, cv_summary_display, sections_json, section_objs_json, saved_threads_dropdown]
        ).then(lambda tid: tid, inputs=st_thread_id, outputs=thread_id_display)

        user_box.submit(
            on_message,
            inputs=[user_box, chat_display, st_chat_history_api, st_thread_id, st_latest_state_obj],
            outputs=[chat_display, st_chat_history_api, st_latest_state_obj, cv_summary_display, sections_json, section_objs_json, user_box]
        )
        send_btn.click(
            on_message,
            inputs=[user_box, chat_display, st_chat_history_api, st_thread_id, st_latest_state_obj],
            outputs=[chat_display, st_chat_history_api, st_latest_state_obj, cv_summary_display, sections_json, section_objs_json, user_box]
        )
        clear_btn.click(
            on_clear,
            outputs=[chat_display, st_chat_history_api, st_latest_state_obj, cv_summary_display, sections_json, section_objs_json, status_display]
        )

        save_state_btn.click(on_save_state, inputs=[st_thread_id, st_latest_state_obj], outputs=[status_display])
        refresh_threads_btn.click(on_refresh_threads, outputs=[saved_threads_dropdown])
        
        load_thread_btn.click(
            on_load_thread,
            inputs=[saved_threads_dropdown],
            outputs=[st_thread_id, chat_display, st_chat_history_api, st_latest_state_obj, cv_summary_display, sections_json, section_objs_json, status_display]
        ).then(lambda tid: tid, inputs=st_thread_id, outputs=thread_id_display)

    return demo

if __name__ == "__main__":
    app = build_api_gradio_ui()
    app.queue().launch()
