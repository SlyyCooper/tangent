from typing import Dict, Optional, List
from ..types import Agent, Structured_Result
import sys

def create_transfer_functions(managed_agents: Dict[str, Agent], triage_agent: Agent) -> List:
    """Create transfer functions for each managed agent"""
    functions = []
    
    for agent_name, agent in managed_agents.items():
        def make_transfer(target_agent):
            def transfer():
                """Transfer to specialized agent"""
                return target_agent
            transfer.__name__ = f"transfer_to_{agent_name.lower()}"
            return transfer
            
        functions.append(make_transfer(agent))
        
        # Add transfer back function to the agent if not already present
        def transfer_back():
            """Return to triage agent"""
            return triage_agent
        transfer_back.__name__ = f"transfer_back_to_{triage_agent.name.lower().replace(' ', '_')}"
        
        if transfer_back.__name__ not in [f.__name__ for f in agent.functions]:
            agent.functions.append(transfer_back)
    
    return functions

def discover_assigned_agents(triage_agent_name: str) -> Dict[str, Agent]:
    """
    Discover all agents that are assigned to this triage agent.
    Searches through all modules in memory for Agent instances.
    """
    assigned_agents = {}
    
    # Make a copy of sys.modules to avoid runtime changes
    modules = dict(sys.modules)
    
    # Look through all modules in memory
    for name, module in modules.items():
        if module is None:
            continue
            
        # Look for Agent instances in the module
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, Agent) 
                    and hasattr(attr, 'triage_assignment')
                    and attr.triage_assignment == triage_agent_name
                ):
                    agent_id = attr_name.lower()
                    assigned_agents[agent_id] = attr
            except:
                continue
    
    return assigned_agents

def enhance_instructions(base_instructions: str, managed_agents: Dict[str, Agent]) -> str:
    """Enhance instructions with managed agent information"""
    agent_descriptions = "\n".join([
        f"- {agent.name}: {agent.instructions if isinstance(agent.instructions, str) else 'Dynamic instructions'}"
        for agent in managed_agents.values()
    ])
    
    return f"""{base_instructions}

Available specialized agents:
{agent_descriptions}

When a user's request matches a specialized agent's domain, transfer to that agent.
If unsure, ask clarifying questions before transferring."""

def create_triage_agent(
    name: str = "Triage Agent",
    instructions: str = "Analyze and route requests to specialized agents",
    managed_agents: Dict[str, Agent] = None,
    auto_discover: bool = True,
    model: str = "gpt-4o"
) -> Agent:
    """
    Create a triage agent that can manage and orchestrate multiple agents.
    """
    # Initialize managed agents
    managed_agents = managed_agents or {}
    
    # Auto-discover assigned agents
    if auto_discover:
        discovered = discover_assigned_agents(name)
        managed_agents.update(discovered)
    
    # Create the triage agent first
    triage_agent = Agent(
        name=name,
        model=model,
        instructions=enhance_instructions(instructions, managed_agents),
        functions=[]  # Start with empty functions
    )
    
    # Now create transfer functions with the agent instance
    triage_agent.functions = create_transfer_functions(managed_agents, triage_agent)
    
    return triage_agent 