import subprocess
from typing import Optional
from tangent.types import Result


def run_applescript(script: str) -> Result:
    """Execute an AppleScript and return its output.
    
    Args:
        script: The AppleScript code to execute
        
    Returns:
        Result object with the script's output and execution status
    """
    try:
        process = subprocess.Popen(
            ['osascript', '-e', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return Result(
                value=f"Error executing AppleScript: {stderr}",
                context_variables={"success": False}
            )
        
        return Result(
            value=stdout.strip(),
            context_variables={"success": True}
        )
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            context_variables={"success": False}
        )


def create_applescript(
    script_name: str,
    script_content: str,
    save_path: Optional[str] = None
) -> Result:
    """Create and save an AppleScript file.
    
    Args:
        script_name: Name of the script (without .scpt extension)
        script_content: The AppleScript code
        save_path: Optional path to save the script. If None, saves to default location
        
    Returns:
        Result object with the path of the saved script
    """
    if not save_path:
        save_path = f"~/Library/Scripts/{script_name}.scpt"
    
    save_path = save_path.replace("~", subprocess.getoutput("echo $HOME"))
    
    try:
        # Convert the script to a compiled format
        process = subprocess.Popen(
            ['osacompile', '-o', save_path],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        _, stderr = process.communicate(input=script_content)
        
        if process.returncode != 0:
            return Result(
                value=f"Error creating AppleScript: {stderr}",
                context_variables={"script_path": None, "success": False}
            )
        
        return Result(
            value=f"Successfully created AppleScript at {save_path}",
            context_variables={"script_path": save_path, "success": True}
        )
        
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            context_variables={"script_path": None, "success": False}
        )


def execute_saved_script(script_path: str) -> Result:
    """Execute a saved AppleScript file.
    
    Args:
        script_path: Path to the .scpt file
        
    Returns:
        Result object with the script's output
    """
    try:
        process = subprocess.Popen(
            ['osascript', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return Result(
                value=f"Error executing script: {stderr}",
                context_variables={"success": False}
            )
            
        return Result(
            value=stdout.strip(),
            context_variables={"success": True}
        )
        
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            context_variables={"success": False}
        )
