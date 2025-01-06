from tangent.types import Structured_Result
import datetime
from zoneinfo import ZoneInfo

def current_time() -> Structured_Result:
    """
    Get the current time in a friendly format (Eastern Time).
    
    Returns:
        Structured_Result object containing the current time
    """
    try:
        # Get current time in Eastern Time
        eastern_time = datetime.datetime.now(ZoneInfo("America/New_York"))
        current = eastern_time.strftime("%I:%M %p")
        
        return Structured_Result(
            value=f"The current time is {current}",
            extracted_data={
                "current_time": current
            }
        )
        
    except Exception as e:
        return Structured_Result(
            value=f"Error getting current time: {str(e)}"
        )
