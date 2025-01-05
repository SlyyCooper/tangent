import subprocess
import datetime
from typing import List, Optional, Union
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    title: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    calendar_name: str
    location: Optional[str] = None
    notes: Optional[str] = None

def _run_calendar_script(start_date: datetime.date, end_date: datetime.date) -> str:
    """Run AppleScript to get calendar events between dates."""
    script = f'''
    tell application "Calendar"
        set eventList to ""
        set startDate to date "{start_date.strftime('%Y-%m-%d')}"
        set endDate to date "{end_date.strftime('%Y-%m-%d')}"
        
        repeat with calendarAccount in calendars
            set eventList to eventList & (name of calendarAccount) & "\\n"
            set evs to every event of calendarAccount whose start date is greater than or equal to startDate and start date is less than or equal to endDate
            
            repeat with ev in evs
                set eventList to eventList & "EVENT_START\\n"
                set eventList to eventList & "title: " & summary of ev & "\\n"
                set eventList to eventList & "calendar: " & (name of calendarAccount) & "\\n"
                set eventList to eventList & "start: " & ((start date of ev) as string) & "\\n"
                set eventList to eventList & "end: " & ((end date of ev) as string) & "\\n"
                
                try
                    set eventList to eventList & "location: " & location of ev & "\\n"
                end try
                
                try
                    set eventList to eventList & "description: " & description of ev & "\\n"
                end try
                
                set eventList to eventList & "EVENT_END\\n"
            end repeat
        end repeat
        
        return eventList
    end tell
    '''
    
    cmd = ['osascript', '-e', script]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Error running Calendar script: {result.stderr}")
    
    return result.stdout

def _parse_calendar_output(output: str) -> List[CalendarEvent]:
    """Parse the output from Calendar AppleScript into CalendarEvent objects."""
    events = []
    current_event = {}
    in_event = False
    
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
            
        if line == "EVENT_START":
            current_event = {}
            in_event = True
            continue
        elif line == "EVENT_END":
            if current_event:
                events.append(CalendarEvent(**current_event))
            in_event = False
            continue
            
        if in_event and ": " in line:
            key, value = line.split(": ", 1)
            if key == "title":
                current_event["title"] = value
            elif key == "calendar":
                current_event["calendar_name"] = value
            elif key == "start":
                current_event["start_time"] = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S +0000")
            elif key == "end":
                current_event["end_time"] = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S +0000")
            elif key == "location":
                current_event["location"] = value
            elif key == "description":
                current_event["notes"] = value
    
    return sorted(events, key=lambda x: x.start_time)

def get_calendar_events(date_spec: str = "today") -> List[CalendarEvent]:
    """
    Get calendar events for a specified date or date range.
    
    Args:
        date_spec (str): Date specification. Can be:
            - "today"
            - "tomorrow"
            - "this week"
            - "next week"
            - A specific date in "YYYY-MM-DD" format
            - A date range in "YYYY-MM-DD,YYYY-MM-DD" format
    
    Returns:
        List[CalendarEvent]: List of calendar events matching the date specification
    """
    try:
        today = datetime.date.today()
        
        if date_spec == "today":
            start_date = end_date = today
        elif date_spec == "tomorrow":
            start_date = end_date = today + datetime.timedelta(days=1)
        elif date_spec == "this week":
            start_date = today
            end_date = today + datetime.timedelta(days=7)
        elif date_spec == "next week":
            start_date = today + datetime.timedelta(days=7)
            end_date = start_date + datetime.timedelta(days=7)
        elif ',' in date_spec:  # Date range
            start_date_str, end_date_str = date_spec.split(',')
            start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        else:  # Single date
            start_date = end_date = datetime.datetime.strptime(date_spec, "%Y-%m-%d").date()
        
        output = _run_calendar_script(start_date, end_date)
        return _parse_calendar_output(output)
        
    except Exception as e:
        raise RuntimeError(f"Failed to get calendar events: {str(e)}")

if __name__ == "__main__":
    # Example usage
    print("\nChecking today's events:")
    events = get_calendar_events("today")
    for event in events:
        print(f"\nEvent: {event.title}")
        print(f"Calendar: {event.calendar_name}")
        print(f"When: {event.start_time.strftime('%I:%M %p')} - {event.end_time.strftime('%I:%M %p')}")
        if event.location:
            print(f"Location: {event.location}")
        if event.notes:
            print(f"Notes: {event.notes}")
            
    print("\nChecking next week's events:")
    events = get_calendar_events("next week")
    for event in events:
        print(f"\nEvent: {event.title}")
        print(f"Calendar: {event.calendar_name}")
        print(f"When: {event.start_time.strftime('%Y-%m-%d %I:%M %p')} - {event.end_time.strftime('%I:%M %p')}")
        if event.location:
            print(f"Location: {event.location}")
        if event.notes:
            print(f"Notes: {event.notes}")
