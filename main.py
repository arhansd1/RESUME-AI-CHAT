import asyncio
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from graph_builder import build_resume_graph, run_gradio_ui
from agents.resume_builder_state import ResumeBuilderState
import asyncio
import logging
import json
import argparse
# Configure logging for the application
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Import the graph builder functions (use absolute import so script can be run directly)
from graph_builder import build_resume_graph, run_interactive_session, run_gradio_ui
from api_gradio_ui import build_api_gradio_ui
from agents.resume_builder_state import ResumeBuilderState

def main():
    """Main function to build and run the resume builder graph."""
    parser = argparse.ArgumentParser(description="Run the Resume AI Chat application.")
    parser.add_argument(
        "--ui",
        type=str,
        choices=["terminal", "gradio", "api_gradio"],
        default="gradio",
        help="Choose the user interface to run: 'terminal', 'gradio' for direct graph interaction, or 'api_gradio' for API-based interaction."
    )
    args = parser.parse_args()
    initial_state = ResumeBuilderState()

    try:
        with open("initial_state.json", "r") as f:
            data = json.load(f)
            initial_state.resume_schema = data.get("resume_schema")
            initial_state.resume_sections = data.get("resume_sections")
            initial_state.jd_summary = data.get("jd_summary")
            initial_state.section_objects = data.get("section_objects", {}) # Load section_objects as well
    except FileNotFoundError:
        logger.error("initial_state.json not found. Please ensure the file exists.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding initial_state.json: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading initial state: {e}")
        return

    # print(f"Initial Resume Schema loaded: {json.dumps(initial_state.resume_schema, indent=2)}")
    print("Initial Resume Sections and JD Summary loaded.")

    if args.ui == "gradio":
            compiled_graph = build_resume_graph(initial_state=initial_state)
            run_gradio_ui(compiled_graph)
    elif args.ui == "api_gradio":
        # The API Gradio UI doesn't need a pre-compiled graph.
        # It interacts with the FastAPI server, which has its own graph instance.
        api_ui = build_api_gradio_ui(initial_state=initial_state)
        api_ui.queue().launch()
    elif args.ui == "terminal":
        compiled_graph = build_resume_graph(initial_state=initial_state)
        # The terminal runner is async, so we need to run it in an event loop.
        asyncio.run(run_interactive_session(compiled_graph))
    else:
        print(f"Unknown UI choice: {args.ui}")



#     compiled_graph = build_resume_graph(initial_state=initial_state)

#     run_gradio_ui(compiled_graph)

if __name__ == "__main__":
    main()
