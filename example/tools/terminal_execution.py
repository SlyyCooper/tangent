from tangent.types import Result
import subprocess
import os

def execute_terminal(command: str) -> Result:
    """
    Execute any terminal command and return the result.
    
    Args:
        command: The command to execute
        
    Returns:
        Result object containing command output
    """
    try:
        # Execute command in shell
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Format output
        output = (
            f"$ {command}\n\n"
            f"{process.stdout}"
        )
        
        if process.stderr:
            output += f"\n{process.stderr}"
            
        return Result(
            value=output,
            extracted_data={
                "command": command,
                "exit_code": process.returncode
            }
        )
        
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            extracted_data={"command": command}
        )
