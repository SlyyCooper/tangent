from tangent import Agent, tangent, run_tangent_loop, process_and_print_streaming_response
from tangent import Result
import os
import subprocess

def run_applescript(script: str) -> Result:
    """
    Execute an AppleScript command and return the results.
    
    Args:
        script: The AppleScript command to execute
    """
    try:
        # Execute the AppleScript command using osascript
        process = subprocess.Popen(['osascript', '-e', script], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8')
            return Result(value=f"Error executing AppleScript: {error_msg}")
        
        # Return successful result
        output = stdout.decode('utf-8').strip()
        return Result(
            value=output,
            context_variables={"last_script": script}
        )
    except Exception as e:
        return Result(value=f"Error executing AppleScript: {str(e)}")

# Create the AppleScript agent
applescript_agent = Agent(
    name="AppleScript Agent",
    model="gpt-4o",
    instructions="""You are a helpful agent that can execute AppleScript commands on macOS.
You can use the run_applescript function to execute AppleScript commands.
Always analyze the command results and provide a concise, informative response.
If a command fails, try to suggest corrections or alternatives.

Here are the exact AppleScript commands to use for common tasks:

1. Notifications:
   - Display notification: 'display notification "Hello World" with title "Title"'

2. System Controls:
   - Set brightness: 'tell application "System Events" to tell process "SystemUIServer" to set value of first slider of first group of first window to 0.5'
   - Get brightness: 'tell application "System Events" to tell process "SystemUIServer" to get value of first slider of first group of first window'
   - Set volume: 'set volume output volume 50'
   - Get volume: 'output volume of (get volume settings)'

3. Applications:
   - Open app: 'tell application "AppName" to activate'
   - Quit app: 'tell application "AppName" to quit'
   - Is app running: 'tell application "System Events" to (name of processes) contains "AppName"'

4. Clipboard:
   - Get clipboard: 'the clipboard'
   - Set clipboard: 'set the clipboard to "text"'

Always use these exact commands - they are tested and working. If a command fails, check the error message and try to diagnose the specific issue.""",
    functions=[run_applescript],
    triage_assignment="System Assistant"  # Assign to System Assistant triage agent
)

# Initialize tangent client
client = tangent()

def run_applescript_conversation(script: str, context_variables: dict = None):
    """
    Run an AppleScript conversation with the agent.
    """
    if context_variables is None:
        context_variables = {}
    
    messages = [{"role": "user", "content": script}]
    
    response = client.run(
        agent=applescript_agent,
        messages=messages,
        context_variables=context_variables,
        stream=True,
        debug=True
    )
    
    return response

if __name__ == "__main__":
    # Run the interactive demo loop
    run_tangent_loop(applescript_agent, stream=True, debug=False) 