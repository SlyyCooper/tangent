import inspect
from datetime import datetime
import os
from pathlib import Path
from typing import Optional


def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def load_instructions_from_file(agent_name: str, instructions_dir: str = "instructions") -> Optional[str]:
    """
    Load instructions from a file in the specified directory.
    Looks for files with the agent's name and either .md or .txt extension.
    
    Args:
        agent_name: Name of the agent (used as filename)
        instructions_dir: Directory to look for instruction files
        
    Returns:
        The contents of the instruction file if found, None otherwise
    """
    # Create Path object for the directory
    dir_path = Path(instructions_dir)
    
    # Check both .md and .txt files
    for ext in [".md", ".txt"]:
        file_path = dir_path / f"{agent_name}{ext}"
        if file_path.exists():
            return file_path.read_text().strip()
    
    return None


def get_instructions(agent) -> str:
    """
    Get instructions for an agent based on its configuration.
    Handles inline strings, callables, and file-based instructions.
    
    Args:
        agent: The Agent instance
        
    Returns:
        The resolved instructions as a string
        
    Raises:
        ValueError: If instructions cannot be loaded
    """
    # Handle inline string
    if agent.instructions_source == "inline":
        if isinstance(agent.instructions, str):
            return agent.instructions
        raise ValueError(f"Invalid inline instructions for agent {agent.name}")
    
    # Handle callable
    if agent.instructions_source == "callable":
        if callable(agent.instructions):
            return agent.instructions()
        raise ValueError(f"Invalid callable instructions for agent {agent.name}")
    
    # Handle file-based
    if agent.instructions_source == "file":
        instructions = load_instructions_from_file(agent.name, agent.instructions_dir)
        if instructions is not None:
            return instructions
        raise ValueError(
            f"Could not find instructions file for agent {agent.name} "
            f"in directory {agent.instructions_dir}"
        )
    
    raise ValueError(f"Unknown instructions source: {agent.instructions_source}")