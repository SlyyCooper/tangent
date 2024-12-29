from tangent import Tangent
from .agents import (
    gateway_agent,
    applescript_agent,
    command_agent,
    TaskContext,
    assess_task_requirements
)
from .evals_util import evaluate_with_llm_bool, BoolEvalResult
import pytest
import json

client = Tangent()

GATEWAY_EVAL_SYSTEM_PROMPT = """
You will evaluate a conversation between a user and a gateway agent that manages Mac OS automation tasks.
Your goal is to assess if the gateway agent effectively manages both user interaction and task orchestration by:

1. Conversation Management
   - Maintains engaging and natural conversation flow
   - Demonstrates context awareness
   - Provides clear and helpful responses
   - Handles multi-turn interactions effectively

2. Task Orchestration
   - Correctly analyzes and breaks down complex requests
   - Chooses appropriate specialized agents
   - Maintains state across agent transitions
   - Handles task dependencies appropriately

3. User Experience
   - Provides consistent interaction patterns
   - Maintains conversation continuity during transitions
   - Gives appropriate feedback and status updates
   - Handles errors and edge cases gracefully

4. Technical Execution
   - Correctly uses AppleScript for GUI automation
   - Appropriately uses Terminal commands for system operations
   - Maintains context and state throughout the session
   - Successfully completes the requested tasks
"""

def evaluate_conversation(messages, prompt=GATEWAY_EVAL_SYSTEM_PROMPT) -> BoolEvalResult:
    """Evaluate a conversation using the specified evaluation prompt"""
    conversation = f"CONVERSATION: {json.dumps(messages)}"
    return evaluate_with_llm_bool(prompt, conversation)

def run_and_get_tool_calls(agent, query, context_data=None):
    """Run an agent with a query and optional context data"""
    message = {"role": "user", "content": query}
    context = {}
    if context_data:
        context = {"context_variables": context_data}
    
    response = client.run(
        agent=agent,
        messages=[message],
        execute_tools=False,
        **context
    )
    return response.messages[-1].get("tool_calls")

@pytest.mark.parametrize(
    "query,expected_agent",
    [
        ("Open System Preferences and enable Dark Mode", "applescript"),
        ("Show me all hidden files in my Downloads folder", "command"),
        ("Create a new reminder in Reminders app", "applescript"),
        ("Check available disk space", "command"),
    ],
)
def test_task_requirement_assessment(query, expected_agent):
    """Test the gateway's task assessment logic"""
    result = assess_task_requirements(query)
    assert result == expected_agent

@pytest.mark.parametrize(
    "query,expected_tools",
    [
        (
            "I need to clean up my desktop and set a reminder",
            ["execute_command", "execute_applescript"]
        ),
        (
            "Create a backup of my documents and add it to my calendar",
            ["execute_command", "execute_applescript"]
        ),
    ],
)
def test_gateway_task_orchestration(query, expected_tools):
    """Test the gateway agent's ability to orchestrate multiple tasks"""
    tool_calls = run_and_get_tool_calls(gateway_agent, query)
    
    assert len(tool_calls) == len(expected_tools)
    for call, expected in zip(tool_calls, expected_tools):
        assert call["function"]["name"] == expected

@pytest.mark.parametrize(
    "conversation_flow",
    [
        # Test complex multi-agent task with context
        [
            {"role": "user", "content": "I need to organize my downloads folder and create a calendar event for tomorrow"},
            {"role": "assistant", "content": "I'll help you with both tasks. Let's break this down:"},
            {"role": "tool", "tool_name": "execute_command", "context": {"folder": "downloads"}},
            {"role": "assistant", "content": "I've organized your downloads folder. Now, let's create that calendar event."},
            {"role": "tool", "tool_name": "execute_applescript", "context": {"date": "tomorrow"}},
            {"role": "assistant", "content": "I've completed both tasks for you. Your downloads are organized and the calendar event is created."},
        ],
        # Test context-aware interaction
        [
            {"role": "user", "content": "Check my battery status"},
            {"role": "tool", "tool_name": "execute_applescript"},
            {"role": "assistant", "content": "Your battery is at 80%. Would you like me to notify you when it's low?"},
            {"role": "user", "content": "Yes, please"},
            {"role": "assistant", "content": "I'll set up a notification for low battery."},
            {"role": "tool", "tool_name": "execute_applescript", "context": {"notification_threshold": "20%"}},
        ],
    ],
)
def test_gateway_conversation_flow(conversation_flow):
    """Test the gateway agent's conversation management and context awareness"""
    result = evaluate_conversation(conversation_flow)
    assert result.value == True
    assert result.reason is not None  # Ensure we get explanation for the evaluation

def test_context_maintenance():
    """Test the gateway agent's ability to maintain context across interactions"""
    test_context = TaskContext()
    
    # Test task management
    test_context.update_task("organize_files")
    assert test_context.current_task == "organize_files"
    assert not test_context.task_history  # First task shouldn't be in history
    
    test_context.update_task("create_calendar_event")
    assert "organize_files" in test_context.task_history
    assert test_context.current_task == "create_calendar_event"
    
    # Test agent tracking
    test_context.track_agent("command")
    test_context.track_agent("applescript")
    assert "command" in test_context.active_agents
    assert "applescript" in test_context.active_agents
    
    # Test state management
    test_context.update_state("file_count", 10)
    test_context.update_state("calendar_date", "tomorrow")
    assert test_context.conversation_state["file_count"] == 10
    assert test_context.conversation_state["calendar_date"] == "tomorrow"

@pytest.mark.parametrize(
    "context_updates",
    [
        (
            [
                ("update_task", "task1"),
                ("track_agent", "command"),
                ("update_state", ("key1", "value1")),
            ]
        ),
        (
            [
                ("update_task", "task2"),
                ("track_agent", "applescript"),
                ("update_state", ("key2", "value2")),
                ("update_task", "task3"),
            ]
        ),
    ],
)
def test_context_operations(context_updates):
    """Test various context operations and state transitions"""
    test_context = TaskContext()
    
    for operation, *args in context_updates:
        if operation == "update_task":
            test_context.update_task(args[0])
        elif operation == "track_agent":
            test_context.track_agent(args[0])
        elif operation == "update_state":
            key, value = args[0]
            test_context.update_state(key, value)
    
    # Verify final state
    if test_context.task_history:
        assert len(test_context.task_history) == len([u for u, *_ in context_updates if u == "update_task"]) - 1
    if test_context.active_agents:
        assert len(test_context.active_agents) == len([u for u, *_ in context_updates if u == "track_agent"])
    if test_context.conversation_state:
        assert len(test_context.conversation_state) == len([u for u, *_ in context_updates if u == "update_state"])