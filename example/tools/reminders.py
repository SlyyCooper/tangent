import subprocess
from datetime import datetime

def create_reminder(name: str, list_name: str = "Reminders", due_date: str | datetime = None) -> str:
    """
    Create a reminder with a name and optional due date.
    
    Args:
        name: The name/title of the reminder
        list_name: The name of the list to create the reminder in (default: "Reminders")
        due_date: Optional due date and time (can be datetime object or string like "tomorrow at 10:00 AM")
    
    Returns:
        The ID of the created reminder
    """
    script = f'''
    tell application "Reminders"
        tell list "{list_name}"'''
    
    if due_date:
        if isinstance(due_date, str):
            # Use AppleScript to parse the relative date
            script += f'''
            set dueDate to (current date)
            if "{due_date}" starts with "tomorrow" then
                set dueDate to dueDate + (24 * 60 * 60) -- add 24 hours
                
                -- Extract time if provided (e.g., "tomorrow at 10:00 AM")
                if "{due_date}" contains " at " then
                    set timeStr to text ((offset of " at " in "{due_date}") + 4) thru -1 of "{due_date}"
                    set hours to (first word of timeStr as number)
                    if timeStr ends with "PM" and hours is less than 12 then
                        set hours to hours + 12
                    end if
                    if timeStr ends with "AM" and hours is equal to 12 then
                        set hours to 0
                    end if
                    set minutes to 0
                    if timeStr contains ":" then
                        set minutes to (text ((offset of ":" in timeStr) + 1) thru ((offset of ":" in timeStr) + 2) of timeStr) as number
                    end if
                    set time of dueDate to (hours * 60 * 60 + minutes * 60)
                end if
            end if
            make new reminder with properties {{name:"{name}", due date:dueDate}}'''
        else:
            # If it's a datetime, format it
            date_str = due_date.strftime("%m/%d/%Y %I:%M:%S %p")
            script += f'''
            make new reminder with properties {{name:"{name}", due date:date "{date_str}"}}'''
    else:
        script += f'''
            make new reminder with properties {{name:"{name}"}}'''
    
    script += '''
        end tell
    end tell
    '''
    
    try:
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception(f"AppleScript error: {e.stderr}")

def get_lists() -> list[str]:
    """Get all reminder list names."""
    script = 'tell application "Reminders" to get name of lists'
    try:
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip().split(", ")
    except subprocess.CalledProcessError as e:
        raise Exception(f"AppleScript error: {e.stderr}")

def get_reminders(list_name: str = "Reminders") -> list[dict]:
    """
    Get all reminders from a specific list.
    Returns a list of dictionaries containing reminder details including due dates.
    """
    script = f'''
    tell application "Reminders"
        tell list "{list_name}"
            set output to ""
            repeat with r in reminders
                set theName to name of r
                set isCompleted to completed of r
                
                try
                    set dueDate to due date of r
                    if dueDate is missing value then
                        set dueDateStr to "none"
                    else
                        set dueDateStr to ((short date string of dueDate) & " at " & ¬
                                         (time string of dueDate))
                    end if
                on error
                    set dueDateStr to "none"
                end try
                
                set output to output & theName & "|" & (isCompleted as string) & "|" & dueDateStr & "\n"
            end repeat
            return output
        end tell
    end tell
    '''
    
    try:
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        output = result.stdout.strip()
        reminders = []
        
        if output:
            # Each line is "name|completed|due_date"
            for line in output.split("\n"):
                if line:
                    name, completed, due_date = line.split("|")
                    reminder = {
                        "name": name,
                        "completed": completed.lower() == "true",
                        "due_date": None if due_date == "none" else due_date
                    }
                    reminders.append(reminder)
        return reminders
    except subprocess.CalledProcessError as e:
        raise Exception(f"AppleScript error: {e.stderr}")

def update_due_date(reminder_name: str, due_date: str | datetime = None, list_name: str = "Reminders") -> bool:
    """
    Update the due date of a reminder.
    
    Args:
        reminder_name: The name of the reminder to update
        due_date: The new due date and time (can be datetime object, string like "tomorrow at 10:00 AM", or None to remove)
        list_name: The name of the list containing the reminder
    
    Returns:
        True if successful, False if reminder not found
    """
    script = f'''
    tell application "Reminders"
        tell list "{list_name}"
            try
                set theReminder to (first reminder whose name is "{reminder_name}")'''
    
    if due_date:
        if isinstance(due_date, str):
            # Use AppleScript to parse the relative date
            script += f'''
                set dueDate to (current date)
                if "{due_date}" starts with "tomorrow" then
                    set dueDate to dueDate + (24 * 60 * 60) -- add 24 hours
                    
                    -- Extract time if provided (e.g., "tomorrow at 10:00 AM")
                    if "{due_date}" contains " at " then
                        set timeStr to text ((offset of " at " in "{due_date}") + 4) thru -1 of "{due_date}"
                        set hours to (first word of timeStr as number)
                        if timeStr ends with "PM" and hours is less than 12 then
                            set hours to hours + 12
                        end if
                        if timeStr ends with "AM" and hours is equal to 12 then
                            set hours to 0
                        end if
                        set minutes to 0
                        if timeStr contains ":" then
                            set minutes to (text ((offset of ":" in timeStr) + 1) thru ((offset of ":" in timeStr) + 2) of timeStr) as number
                        end if
                        set time of dueDate to (hours * 60 * 60 + minutes * 60)
                    end if
                end if
                set due date of theReminder to dueDate'''
        else:
            # If it's a datetime, format it
            date_str = due_date.strftime("%m/%d/%Y %I:%M:%S %p")
            script += f'''
                set due date of theReminder to date "{date_str}"'''
    else:
        script += '''
                set due date of theReminder to missing value'''
    
    script += '''
                return true
            on error
                return false
            end try
        end tell
    end tell
    '''
    
    try:
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip().lower() == "true"
    except subprocess.CalledProcessError as e:
        raise Exception(f"AppleScript error: {e.stderr}")
