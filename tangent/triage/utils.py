from typing import Dict, List
from ..types import Agent

def discover_agents(module) -> Dict[str, Agent]:
    """
    Discover all Agent instances in a given module.
    
    Args:
        module: The module to search for agents
        
    Returns:
        Dict[str, Agent]: Dictionary of discovered agents
    """
    agents = {}
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, Agent):
            agent_id = attr_name.lower()
            agents[agent_id] = attr
    return agents

def validate_agent_compatibility(agent: Agent) -> bool:
    """
    Validate if an agent is compatible with triage system.
    
    Args:
        agent: The agent to validate
        
    Returns:
        bool: True if agent is compatible
    """
    return (
        hasattr(agent, 'name') and
        hasattr(agent, 'instructions') and
        hasattr(agent, 'functions')
    )

def generate_agent_description(agent: Agent) -> str:
    """
    Generate a human-readable description of an agent's capabilities.
    
    Args:
        agent: The agent to describe
        
    Returns:
        str: Human-readable description
    """
    capabilities = []
    for func in agent.functions:
        if hasattr(func, '__doc__') and func.__doc__:
            capabilities.append(f"- {func.__doc__.strip()}")
    
    return f"""Agent: {agent.name}
Instructions: {agent.instructions if isinstance(agent.instructions, str) else 'Dynamic instructions'}
Capabilities:
{''.join(capabilities)}""" 