import os
import subprocess
from typing import Optional, List, Dict
from tangent.types import Result


def run_command(command: str, cwd: Optional[str] = None) -> Result:
    """Execute a shell command and return its output.
    
    Args:
        command: The shell command to execute
        cwd: Optional working directory for command execution
        
    Returns:
        Result object with command output and status
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return Result(
                value=f"Error: {stderr}",
                context_variables={
                    "success": False,
                    "exit_code": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                }
            )
        
        return Result(
            value=stdout.strip(),
            context_variables={
                "success": True,
                "exit_code": 0,
                "stdout": stdout,
                "stderr": stderr
            }
        )
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e)
            }
        )


def run_background_command(command: str, cwd: Optional[str] = None) -> Result:
    """Execute a command in the background.
    
    Args:
        command: The shell command to execute
        cwd: Optional working directory for command execution
        
    Returns:
        Result object with process information
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=True
        )
        
        return Result(
            value=f"Started background process with PID: {process.pid}",
            context_variables={
                "success": True,
                "pid": process.pid
            }
        )
    except Exception as e:
        return Result(
            value=f"Error starting background process: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e)
            }
        )


def get_environment_info() -> Result:
    """Get information about the current shell environment.
    
    Returns:
        Result object with environment information
    """
    try:
        env_info = {
            "shell": os.environ.get("SHELL", ""),
            "path": os.environ.get("PATH", ""),
            "home": os.environ.get("HOME", ""),
            "user": os.environ.get("USER", ""),
            "pwd": os.getcwd(),
            "python_path": os.environ.get("PYTHONPATH", "")
        }
        
        return Result(
            value=str(env_info),
            context_variables={
                "success": True,
                "environment": env_info
            }
        )
    except Exception as e:
        return Result(
            value=f"Error getting environment info: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e)
            }
        )


def kill_process(pid: int) -> Result:
    """Kill a background process by its PID.
    
    Args:
        pid: Process ID to kill
        
    Returns:
        Result object with operation status
    """
    try:
        os.kill(pid, 15)  # SIGTERM
        return Result(
            value=f"Successfully terminated process {pid}",
            context_variables={
                "success": True,
                "pid": pid
            }
        )
    except ProcessLookupError:
        return Result(
            value=f"Process {pid} not found",
            context_variables={
                "success": False,
                "error": "Process not found"
            }
        )
    except Exception as e:
        return Result(
            value=f"Error killing process {pid}: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e)
            }
        )
