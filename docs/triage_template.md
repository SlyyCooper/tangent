# Triage Agent Template

This template shows the basic structure needed to create a triage agent system that can handle multiple specialized tasks.

# Overview of the Triage Agent
1. How it is Dynamically Aware of the Agents it Can Transfer to:
- Through Function Registration: The triage agent knows about other agents through its `functions` list. In the code, we see:
  ```python
  triage_agent.functions = [transfer_to_agent_1, transfer_to_agent_2]
  ```
- Through Transfer Functions: Each transfer function returns a specific agent instance:
  ```python
  def transfer_to_agent_1():
      return agent_1  # Returns the actual agent instance
  ```
- Through Bidirectional Links: Each specialized agent knows about the triage agent through:
  ```python
  agent_1.functions.append(transfer_back_to_triage)
  ```

2. How it is Able to Hold a Conversation AND Transfer to the Appropriate Agent:
- Through Instructions: The triage agent has specific instructions that tell it how to analyze requests:
  ```python
  instructions="Analyze user requests and transfer to appropriate specialized agent"
  ```
- Through Context: It receives context that helps it make decisions:
  ```python
  context = {
      "context_1": "Information all agents might need",
      "context_2": "More shared information"
  }
  ```
- Through Message Analysis: As seen in the airline example's TRIAGE_SYSTEM_PROMPT:
  ```python
  """You are to triage a users request, and call a tool to transfer to the right intent.
     When you need more information to triage the request to an agent, ask a direct question.
     Do not share your thought process with the user!"""
  ```
- Through Function Calling: When it identifies a need to transfer, it calls the appropriate transfer function
- Through Conversation State: The agent maintains conversation state until it determines a transfer is needed

## Basic Structure

```python
# config/agents.py
from tangent import Agent

# 1. Define the specialized functions each agent can use
def specialized_function_1(param1, param2):
    """What this function does"""
    return "Result"

def specialized_function_2():
    """What this function does"""
    return "Result"

# 2. Define transfer functions
def transfer_to_agent_1():
    """Transfer to Agent 1"""
    return agent_1

def transfer_to_agent_2():
    """Transfer to Agent 2"""
    return agent_2

def transfer_back_to_triage():
    """Return to triage agent"""
    return triage_agent

# 3. Define the agents
triage_agent = Agent(
    name="Triage Agent",
    instructions="Analyze user requests and transfer to appropriate specialized agent",
    functions=[transfer_to_agent_1, transfer_to_agent_2]  # Can transfer to any specialized agent
)

agent_1 = Agent(
    name="Specialized Agent 1",
    instructions="Handle specific type of task 1",
    functions=[
        specialized_function_1,
        transfer_back_to_triage  # Can return to triage
    ]
)

agent_2 = Agent(
    name="Specialized Agent 2",
    instructions="Handle specific type of task 2",
    functions=[
        specialized_function_2,
        transfer_back_to_triage  # Can return to triage
    ]
)
```

```python
# main.py
from tangent.repl import run_tangent_loop
from config.agents import triage_agent

# Optional: Add context that all agents can access
context = {
    "context_1": "Information all agents might need",
    "context_2": "More shared information"
}

if __name__ == "__main__":
    run_tangent_loop(triage_agent, context_variables=context)
```

## Flow Explanation

1. User sends message to triage agent
2. Triage agent analyzes request and either:
   - Handles it directly if within its scope
   - Transfers to appropriate specialized agent using transfer functions
3. Specialized agent either:
   - Handles the request using its specialized functions
   - Transfers back to triage if request is outside its scope
4. Process repeats as needed

## Directory Structure
```
your_project/
├── config/
│   ├── __init__.py
│   ├── agents.py    # Define all agents and their functions
│   └── tools.py     # Optional: Define shared tools/functions
├── main.py          # Entry point
└── README.md
```

## Key Points

1. **Triage Agent**
   - First point of contact
   - Only needs transfer functions
   - Analyzes and routes requests

2. **Specialized Agents**
   - Have specific functions for their domain
   - Can always transfer back to triage
   - Clear, focused responsibilities

3. **Transfer Functions**
   - Simple, return the target agent
   - Create the network between agents
   - Enable bi-directional transfers

4. **Context**
   - Optional shared information
   - Available to all agents
   - Helps inform decision making
``` 