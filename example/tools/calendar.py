import subprocess
import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

class EventStatus(str, Enum):
    NONE = "none"
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"

class CalendarEvent(BaseModel):
    title: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    calendar_name: str
    location: Optional[str] = None
    notes: Optional[str] = None
    url: Optional[str] = Field(None, description="URL for video calls or meeting links")
    attendees: Optional[List[str]] = Field(default_factory=list, description="List of attendee email addresses")
    is_all_day: bool = False
    status: EventStatus = EventStatus.NONE

def _run_calendar_script(start_date: datetime.date, end_date: datetime.date) -> str:
    """Run AppleScript to get calendar events between dates."""
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    script = f'''
    tell application "Calendar"
        set eventList to ""
        set startDate to date "{start_str}"
        set endDate to date "{end_str}"
        
        set allEvents to {{}}
        repeat with calendarAccount in calendars
            tell calendarAccount
                set calendarEvents to (every event whose start date ≥ startDate and start date ≤ endDate)
                if length of calendarEvents > 0 then
                    set eventList to eventList & "CALENDAR:" & name & return
                    repeat with anEvent in calendarEvents
                        set eventList to eventList & "EVENT_START" & return
                        set eventList to eventList & "title:" & summary of anEvent & return
                        set eventList to eventList & "start:" & ((start date of anEvent) as string) & return
                        set eventList to eventList & "end:" & ((end date of anEvent) as string) & return
                        set eventList to eventList & "all_day:" & (allday of anEvent as string) & return
                        if location of anEvent is not missing value then
                            set eventList to eventList & "location:" & location of anEvent & return
                        end if
                        if description of anEvent is not missing value then
                            set eventList to eventList & "description:" & description of anEvent & return
                        end if
                        set eventList to eventList & "EVENT_END" & return
                    end repeat
                end if
            end tell
        end repeat
        return eventList
    end tell
    '''
    
    cmd = ['osascript', '-e', script]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Error running Calendar script: {result.stderr}")
    
    print("Raw Calendar Output:")  # Debug output
    print(result.stdout)  # Debug output
    
    return result.stdout

def _parse_calendar_output(output: str) -> List[CalendarEvent]:
    """Parse the output from Calendar AppleScript into CalendarEvent objects."""
    events = []
    current_event = {}
    current_calendar = None
    in_event = False
    
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("CALENDAR:"):
            current_calendar = line.split(":", 1)[1]
            continue
            
        if line == "EVENT_START":
            current_event = {"calendar_name": current_calendar}
            in_event = True
            continue
        elif line == "EVENT_END":
            if current_event:
                # Convert date strings to datetime objects
                if 'start' in current_event:
                    try:
                        current_event['start_time'] = datetime.datetime.strptime(
                            current_event.pop('start').split("+")[0], 
                            "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        continue
                if 'end' in current_event:
                    try:
                        current_event['end_time'] = datetime.datetime.strptime(
                            current_event.pop('end').split("+")[0], 
                            "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        continue
                
                # Convert all_day string to boolean
                if 'all_day' in current_event:
                    current_event['is_all_day'] = current_event.pop('all_day').lower() == 'true'
                
                # Move description to notes
                if 'description' in current_event:
                    current_event['notes'] = current_event.pop('description')
                
                try:
                    events.append(CalendarEvent(**current_event))
                except Exception:
                    continue
                    
            in_event = False
            continue
            
        if in_event and ":" in line:
            key, value = line.split(":", 1)
            current_event[key] = value
    
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
        if event.url:
            print(f"Meeting URL: {event.url}")
        if event.attendees:
            print(f"Attendees: {', '.join(event.attendees)}")
        if event.notes:
            print(f"Notes: {event.notes}")
        print(f"Status: {event.status.value}")
        print(f"All Day: {'Yes' if event.is_all_day else 'No'}")
            
    print("\nChecking next week's events:")
    events = get_calendar_events("next week")
    for event in events:
        print(f"\nEvent: {event.title}")
        print(f"Calendar: {event.calendar_name}")
        print(f"When: {event.start_time.strftime('%Y-%m-%d %I:%M %p')} - {event.end_time.strftime('%I:%M %p')}")
        if event.location:
            print(f"Location: {event.location}")
        if event.url:
            print(f"Meeting URL: {event.url}")
        if event.attendees:
            print(f"Attendees: {', '.join(event.attendees)}")
        if event.notes:
            print(f"Notes: {event.notes}")
        print(f"Status: {event.status.value}")
        print(f"All Day: {'Yes' if event.is_all_day else 'No'}")
