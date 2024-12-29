from tangent import Agent
from tangent.types import Result
from config.tools.applescript_tool import run_applescript, create_applescript, execute_saved_script
from config.tools.terminal_tool import run_command, run_background_command, get_environment_info, kill_process


def execute_applescript(script: str) -> Result:
    """Execute an AppleScript on the user's Mac OS computer."""
    return run_applescript(script)


def create_new_applescript(script_name: str, script_content: str, save_path: str = None) -> Result:
    """Create and save a new AppleScript."""
    return create_applescript(script_name, script_content, save_path)


def run_saved_applescript(script_path: str) -> Result:
    """Execute a saved AppleScript file."""
    return execute_saved_script(script_path)


def execute_command(command: str, cwd: str = None) -> Result:
    """Execute a terminal command."""
    return run_command(command, cwd)


def execute_background_command(command: str, cwd: str = None) -> Result:
    """Execute a command in the background."""
    return run_background_command(command, cwd)


def get_shell_info() -> Result:
    """Get information about the current shell environment."""
    return get_environment_info()


def terminate_process(pid: int) -> Result:
    """Kill a background process by its PID."""
    return kill_process(pid)


class TaskContext:
    """Maintains state and context across agent transitions"""
    def __init__(self):
        self.current_task = None
        self.task_history = []
        self.user_preferences = {}
        self.active_agents = set()
        self.conversation_state = {}
    
    def update_task(self, task):
        self.task_history.append(self.current_task)
        self.current_task = task
    
    def track_agent(self, agent_name):
        self.active_agents.add(agent_name)
    
    def update_state(self, key, value):
        self.conversation_state[key] = value


# Create shared context
gateway_context = TaskContext()


def delegate_to_applescript(context_data=None):
    """Enhanced delegation to AppleScript agent with context"""
    gateway_context.track_agent('applescript')
    if context_data:
        gateway_context.update_state('applescript_context', context_data)
    return applescript_agent


def delegate_to_command(context_data=None):
    """Enhanced delegation to Command agent with context"""
    gateway_context.track_agent('command')
    if context_data:
        gateway_context.update_state('command_context', context_data)
    return command_agent


def return_to_gateway(task_result=None):
    """Return control to gateway agent with task results"""
    if task_result:
        gateway_context.update_state('last_task_result', task_result)
    return gateway_agent


def assess_task_requirements(task_description):
    """Analyzes task requirements and determines optimal agent allocation"""
    if any(gui_term in task_description.lower() for gui_term in ['click', 'open', 'interface', 'window', 'menu']):
        return 'applescript'
    elif any(cmd_term in task_description.lower() for cmd_term in ['file', 'directory', 'process', 'network']):
        return 'command'
    return 'gateway'


# Define specialized agents
applescript_agent = Agent(
    name="Applescript Agent",
    model="gpt-4o",
    instructions="""You are a specialized agent for Mac OS GUI automation using AppleScript.
    
Your responsibilities:
1. Handle all GUI-based automation tasks
2. Interact with Mac OS applications
3. Manage system preferences and settings
4. Create and execute AppleScript commands effectively
5. Save and manage AppleScript files
6. Return results or control to the gateway agent when done

You can:
- Execute immediate AppleScript commands
- Create and save AppleScript files for later use
- Execute saved AppleScript files
- Handle complex GUI automation tasks""",
    functions=[execute_applescript, create_new_applescript, run_saved_applescript, return_to_gateway]
)


command_agent = Agent(
    name="Command Agent",
    model="gpt-4o",
    instructions="""You are a specialized agent for Mac OS terminal operations.
    
Your responsibilities:
1. Handle all file system operations
2. Execute system utilities and commands
3. Manage processes and services
4. Monitor system resources
5. Execute background processes when needed
6. Return results or control to the gateway agent when done

You can:
- Execute terminal commands
- Run background processes
- Get shell environment information
- Kill background processes
- Handle complex terminal operations""",
    functions=[
        execute_command,
        execute_background_command,
        get_shell_info,
        terminate_process,
        return_to_gateway
    ]
)


gateway_agent = Agent(
    name="Gateway Agent",
    model="gpt-4o",
    instructions="""You are a sophisticated gateway agent that manages both user interaction and task orchestration.

Your responsibilities:
1. Maintain engaging, context-aware conversations with users
2. Analyze and break down complex requests into manageable tasks
3. Manage task delegation to specialized agents while maintaining conversation continuity
4. Track and manage state across agent transitions
5. Ensure consistent user experience throughout the interaction

Before delegating:
- Gather all necessary context from the user
- Validate task requirements
- Consider task dependencies
- Maintain conversation history and state

When delegating:
- Choose the most appropriate specialized agent
- Transfer relevant context and state
- Monitor task execution
- Handle transitions smoothly
- Maintain user engagement""",
    functions=[
        delegate_to_applescript,
        delegate_to_command,
        assess_task_requirements,
        execute_applescript,
        create_new_applescript,
        run_saved_applescript,
        execute_command,
        execute_background_command,
        get_shell_info,
        terminate_process
    ]
)

