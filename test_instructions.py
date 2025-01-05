from tangent import setup_agent, get_user_input, show_ai_response, process_chat
from tangent.types import InstructionsSource

def get_dynamic_instructions(context_variables=None):
    """Dynamic instructions based on time of day."""
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 12:
        return "You are a morning assistant, bright and energetic!"
    elif hour < 18:
        return "You are an afternoon assistant, focused and productive!"
    else:
        return "You are an evening assistant, calm and relaxing!"

# Test 1: Inline Instructions
print("\n=== Testing Inline Instructions ===")
client1, agent1 = setup_agent(
    name="InlineBot",
    model="gpt-4o",
    instructions="You are a simple inline instruction bot."
)
response1 = process_chat(client1, agent1, "What kind of bot are you?", stream=False)
show_ai_response(response1)

# Test 2: Callable Instructions
print("\n=== Testing Callable Instructions ===")
client2, agent2 = setup_agent(
    name="DynamicBot",
    model="gpt-4o",
    instructions=get_dynamic_instructions
)
agent2.instructions_source = InstructionsSource.CALLABLE
response2 = process_chat(client2, agent2, "What time of day are you optimized for?", stream=False)
show_ai_response(response2)

# Test 3: File Instructions
print("\n=== Testing File Instructions ===")
client3, agent3 = setup_agent(
    name="ChatBot",  # Will load from instructions/ChatBot.md
    model="gpt-4o",
    instructions="from_file"
)
agent3.instructions_source = InstructionsSource.FILE
response3 = process_chat(client3, agent3, "What kind of conversations do you excel at?", stream=False)
show_ai_response(response3)

print("\nAll tests completed!")
