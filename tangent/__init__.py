from .core import tangent
from .types import Agent, Response, Structured_Result
from .repl import run_chat_loop, process_and_print_streaming_response, pretty_print_messages  # Legacy display functions
from .triage import create_triage_agent
from .embeddings import DocumentStore
from .tools import search_knowledge_base
from .helpers import setup_agent, get_user_input, show_ai_response, process_chat

# Make everything available at the package level
__all__ = [
    "tangent",
    "Agent",
    "Response",
    "Structured_Result",
    "run_chat_loop",
    "process_and_print_streaming_response",  # Legacy: use show_ai_response instead
    "pretty_print_messages",                 # Legacy: use show_ai_response instead
    "create_triage_agent",
    "DocumentStore",
    "search_knowledge_base",
    "setup_agent",           # ✅ Now supports vision=True
    "get_user_input",
    "show_ai_response",      # ✅ Recommended for all response display
    "process_chat",          # ✅ Now supports image/images parameters
]
