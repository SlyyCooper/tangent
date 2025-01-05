from tangent import Agent, tangent
import datetime
from utils.vector_db import ensure_qdrant_running

# Ensure Qdrant is running before importing tools that need it
try:
    ensure_qdrant_running()
except Exception as e:
    print(f"❌ Error ensuring Qdrant is running: {str(e)}")
    print("⚠️ Semantic search functionality may be limited.")

# Import tools after Qdrant is running
from tools.current_time import current_time
from tools.semantic_search import search_documents
from tools.web_search import web_search, web_extract
from tools.terminal_execution import execute_terminal
from tools.reminders import (
    create_reminder,
    get_reminders,
    get_lists
)
from tools.mac_calendar import get_calendar_events
from tools.image_creator import generate_image

current_day = datetime.datetime.now().strftime("%A")
current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")

# Create the document embedding agent
main_agent = Agent(
    name="MyAgent",
    model="gpt-4o",
    instructions=f"""You are a an assistant for Trevor. You're his bro. Act like a bro bro. But don't let your bro-ness fool anybody, your intelligence is hell. Just like Trevor.
    
    Your current tools are:
    1. search_documents: Search through Trevor's document collection
    2. web_search: Search the web for real-time information
    3. web_extract: Extract content from specific URLs (up to 20 at a time)
    4. execute_terminal: Execute any terminal command Trevor needs
    5. current_time: Get the current time
    6. create_reminder: Create a new reminder in any list
    7. get_reminders: Get all reminders from a specific list (shows both active and completed)
    8. get_lists: Get all available reminder lists
    9. get_calendar_events: Check Trevor's calendar events for any date or date range
        - Can check "today", "tomorrow", "this week", "next week"
        - Can check specific dates like "2025-01-04"
        - Can check date ranges like "2025-01-04,2025-01-05"
    10. generate_image: Generate images using Replicate's Flux model
        - Supports various aspect ratios: 1:1, 3:2, 2:3, 16:9, 9:16
        - Creates high-quality images based on text descriptions

    Today is {current_date}.
    """,
    functions=[
        search_documents,
        web_search,
        web_extract,
        execute_terminal,
        current_time,
        create_reminder,
        get_reminders,
        get_lists,
        get_calendar_events,
        generate_image
    ],
    triage_assignment="MyAgent"
)

if __name__ == "__main__":
    from tangent.repl import run_chat_loop
    import datetime
    
    print(f"\nYo what up Trevor! How's it going on this awesome {current_day}? 🤙\n")
    
    # Run the interactive demo loop
    run_chat_loop(main_agent, stream=True, debug=False)
