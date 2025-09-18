"""Graph construction and interactive runner for the resume assistant.

This module wires together:
  * A routing node (LLM driven) that decides which resume section to enter
  * Simple placeholder section nodes (skills, experiences, education, projects)
  * A lightweight terminal chat loop that supports assistant-first greeting

Readability goals:
  - Eliminate duplicate section node logic via a factory
  - Provide clear constants & helper functions
  - Keep state continuity (context + section objects) while resetting execution metadata each turn
  - Trim chat history to avoid unbounded growth
"""

from typing import Any, Dict
from uuid import uuid4

from pyagenity.graph import StateGraph, CompiledGraph
from pyagenity.utils import Message, END, START, CallbackManager
from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.publisher import ConsolePublisher

from agents.resume_builder_state import ResumeBuilderState
from agents.general_chat_section_routing import general_chat_and_section_routing
from agents.create_section_node import create_section_node
from agents.utils import (
    logger,
    SECTION_NAMES,
    MAX_MESSAGES,
    RECURSION_LIMIT,
    last_assistant_message,
    clone_state,
    format_ai_with_section,
    extract_assistant_text,
)

#checkpointer
checkpointer = InMemoryCheckpointer[ResumeBuilderState]()

def build_resume_graph(
    checkpointer: InMemoryCheckpointer[ResumeBuilderState] | None = None,
    publisher: ConsolePublisher | None = None,
    # dependency_container: DependencyContainer | None = None,
    callback_manager: CallbackManager | None = None,
    initial_state: ResumeBuilderState | None = None,
) -> CompiledGraph[ResumeBuilderState]:
    """Build & compile the resume assistant graph with routing + section nodes.

    Parameters:
        initial_state: If provided, this pre-populated state (with jd_summary, section_objects,
            resume_sections, etc.) is used instead of a fresh blank state.
    """
    print("Building the resume builder graph...")

    checkpointer = checkpointer or InMemoryCheckpointer[ResumeBuilderState]()
    publisher = publisher or ConsolePublisher()
    # dependency_container = dependency_container or DependencyContainer()
    callback_manager = callback_manager or CallbackManager()

    graph = StateGraph[ResumeBuilderState](
        state=initial_state or ResumeBuilderState(),
        publisher=publisher,
        # dependency_container=dependency_container,
    )

    graph.add_node("AnalyzeUserQuery", general_chat_and_section_routing)
    
    # Dynamically add section nodes
    for section_name in SECTION_NAMES:
        node_name = f"{section_name.capitalize()}Section"
        graph.add_node(node_name, create_section_node(section_name))

    graph.set_entry_point("AnalyzeUserQuery")

    def route_to_section(state: ResumeBuilderState, **_: Any) -> str:
        if getattr(state, "src_node", None) == state.current_section:
            return END
        if state.current_section is None:
            return "general_chat"

        if state.current_section in SECTION_NAMES:
            print(f"Routing to section: {state.current_section}")
            return state.current_section
        return END

    # Add conditional edges for initial routing from AnalyzeUserQuery
    routing_map = {section_name: f"{section_name.capitalize()}Section" for section_name in SECTION_NAMES}
    routing_map[END] = END
    graph.add_conditional_edges("AnalyzeUserQuery", route_to_section, routing_map)

    # Allow in-section follow-up routing or termination.
    for section_name in SECTION_NAMES:
        node_name = f"{section_name.capitalize()}Section"
        section_routing_map = {s_name: f"{s_name.capitalize()}Section" for s_name in SECTION_NAMES}
        section_routing_map["general_chat"] = "AnalyzeUserQuery"
        section_routing_map[END] = END
        graph.add_conditional_edges(node_name, route_to_section, section_routing_map)

    print("Compiling the graph...")
    compiled = graph.compile(checkpointer=checkpointer)
    print("Graph compiled successfully.")
    return compiled


#graph
app = build_resume_graph(checkpointer=checkpointer)

# --- Interactive Terminal Chat Runner ---
async def run_interactive_session(compiled_graph: CompiledGraph[ResumeBuilderState]):
    """Interactive terminal loop for the compiled graph."""
    print("\nWelcome to the Resume Builder Chat!")
    print("Type 'quit' or 'exit' to end the session.")
    print("Ask about your JD, resume, or route to sections: skills, experiences, education, projects.")

    thread_id = str(uuid4())
    base_config = {"thread_id": thread_id, "recursion_limit": RECURSION_LIMIT}
    current_input: Dict[str, Any] = {"messages": [Message.from_text("SESSION_START", role="system")]}

    try:
        first = await compiled_graph.ainvoke(current_input, base_config, response_granularity="full")
        current_input["messages"] = first.get("messages", [])
        if (st := first.get("state")):
            current_input["state"] = clone_state(st)
        print(f"AI: {extract_assistant_text(first)}")
    except Exception as e:
        logger.error(f"Startup error: {e}")

    while True:
        user_text = input("You: ")
        if user_text.lower() in {"quit", "exit"}:
            break

        current_input["messages"].append(Message.from_text(user_text, role="user"))
        logger.debug(f"all messages: {[m.content for m in current_input['messages']]}")

        if len(current_input["messages"]) > MAX_MESSAGES:
            current_input["messages"] = current_input["messages"][-MAX_MESSAGES:]
            print("(Trimmed chat history)")
            logger.debug(f"Current message count: {len(current_input['messages'])}")
            logger.debug(f"all messages: {[m.content for m in current_input['messages']]}")
        try:
            result = await compiled_graph.ainvoke(current_input, base_config, response_granularity="full")
            current_input["messages"] = result.get("messages", [])
            if (st := result.get("state")):
                current_input["state"] = clone_state(st)
            print(f"AI: {extract_assistant_text(result)}")
        except Exception as e:
            logger.error(f"Error: {e}")
            break

    print("\nChat session ended. Goodbye!")

def run_gradio_ui(compiled_graph):
    """
    Launch a Gradio UI with:
    - Left: AI chat powered by the compiled graph
    - Right: Live JSON views for resume_sections, section_objects, and CV summary
    """
    import gradio as gr

    thread_id = str(uuid4())
    base_config = {"thread_id": thread_id, "recursion_limit": RECURSION_LIMIT}
    current_input = {"messages": [Message.from_text("SESSION_START", role="system")]}

    with gr.Blocks(title="Resume Builder Chat") as demo:
        gr.Markdown("## Resume Builder Chat")
        gr.Markdown("Chat with the AI on the left. The right panel shows live resume sections, section objects, and CV summary from the current state.")

        st_current_input = gr.State(current_input)
        st_base_config = gr.State(base_config)

        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="AI Chat", height=540)
                with gr.Row():
                    user_box = gr.Textbox(placeholder="Ask about your JD, resume, or route to sections (skills, experiences, education, projects)...", lines=3, show_label=False)
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

            with gr.Column(scale=1):
                gr.Markdown("### CV Summary (live)")
                cv_summary_display = gr.JSON(value={}, label="cv_summary", elem_id="cv_summary_json")
                gr.Markdown("### resume_sections (live)")
                sections_json = gr.JSON(value={}, label="resume_sections", elem_id="resume_sections_json")
                gr.Markdown("### section_objects (live)")
                section_objs_json = gr.JSON(value={}, label="section_objects", elem_id="section_objects_json")

        async def on_load(current_input, base_config):
            try:
                first = await compiled_graph.ainvoke(current_input, base_config, response_granularity="full")
                current_input["messages"] = first.get("messages", [])
                if (st := first.get("state")):
                    current_input["state"] = clone_state(st)
                
                st_obj = current_input.get("state")
                ai_text = extract_assistant_text(first)
                ai_text = format_ai_with_section(ai_text, st_obj)
                chat_history = [("", ai_text)]

                resume_sections = getattr(st_obj, "resume_sections", {}) or {}
                section_objects = getattr(st_obj, "section_objects", {}) or {}
                cv_summary = getattr(st_obj, "cv_summary", {}) or {}

                return chat_history, cv_summary, resume_sections, section_objects, current_input
            except Exception as e:
                logger.error(f"Startup error in Gradio on_load: {e}")
                return [("", f"Startup error: {e}")], {}, {}, {}, current_input

        async def on_message(user_text, chat_history, current_input, base_config):
            if not user_text or not user_text.strip():
                st_obj = current_input.get("state")
                return (chat_history, 
                       getattr(st_obj, "cv_summary", {}) or {}, 
                       getattr(st_obj, "resume_sections", {}) or {}, 
                       getattr(st_obj, "section_objects", {}) or {}, 
                       current_input, "")

            current_input["messages"].append(Message.from_text(user_text, role="user"))

            if len(current_input["messages"]) > MAX_MESSAGES:
                current_input["messages"] = current_input["messages"][-MAX_MESSAGES:]
                print("(Trimmed chat history)")

            try:
                result = await compiled_graph.ainvoke(current_input, base_config, response_granularity="full")
                current_input["messages"] = result.get("messages", [])
                if (st := result.get("state")):
                    current_input["state"] = clone_state(st)

                st_obj = current_input.get("state")
                ai_text = extract_assistant_text(result)
                ai_text = format_ai_with_section(ai_text, st_obj)
                chat_history = chat_history + [(user_text, ai_text)]

                resume_sections = getattr(st_obj, "resume_sections", {}) or {}
                section_objects = getattr(st_obj, "section_objects", {}) or {}
                cv_summary = getattr(st_obj, "cv_summary", {}) or {}

                return chat_history, cv_summary, resume_sections, section_objects, current_input, ""
            except Exception as e:
                logger.error(f"Error in Gradio on_message: {e}")
                chat_history = chat_history + [(user_text, f"Error: {e}")]
                st_obj = current_input.get("state")
                return (chat_history, 
                       getattr(st_obj, "cv_summary", {}) or {}, 
                       getattr(st_obj, "resume_sections", {}) or {}, 
                       getattr(st_obj, "section_objects", {}) or {}, 
                       current_input, "")

        def on_clear():
            return [], ""

        demo.load(
            on_load,
            inputs=[st_current_input, st_base_config],
            outputs=[chat, cv_summary_display, sections_json, section_objs_json, st_current_input],
        )

        send_btn.click(
            on_message,
            inputs=[user_box, chat, st_current_input, st_base_config],
            outputs=[chat, cv_summary_display, sections_json, section_objs_json, st_current_input, user_box],
        )
        user_box.submit(
            on_message,
            inputs=[user_box, chat, st_current_input, st_base_config],
            outputs=[chat, cv_summary_display, sections_json, section_objs_json, st_current_input, user_box],
        )
        clear_btn.click(on_clear, inputs=None, outputs=[chat, user_box])

    demo.queue().launch()
