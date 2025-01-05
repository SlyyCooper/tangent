from tangent.types import Result
import datetime
from zoneinfo import ZoneInfo

def current_time() -> Result:
    """
    Get the current time in a friendly format (Eastern Time).
    
    Returns:
        Result object containing the current time
    """
    try:
        # Get current time in Eastern Time
        eastern_time = datetime.datetime.now(ZoneInfo("America/New_York"))
        current = eastern_time.strftime("%I:%M %p")
        
        return Result(
            value=f"The current time is {current}",
            context_variables={
                "current_time": current
            }
        )
        
    except Exception as e:
        return Result(
            value=f"Error getting current time: {str(e)}"
        )
