from tangent import Agent, tangent
from tangent.repl.repl import run_tangent_loop
from tangent.types import Result
from gpt_researcher import GPTResearcher
import asyncio
from typing import Optional, Dict, Any

# Available report types for the researcher
REPORT_TYPES = ["detailed_report", "brief_report", "bullet_points"]

async def conduct_research(query: str, report_type: str = "detailed_report", context_variables: Optional[Dict[str, Any]] = None) -> Result:
    """
    Conduct research on a given query and return a report.
    
    Args:
        query: The research query to investigate
        report_type: Type of report to generate (detailed_report, brief_report, or bullet_points)
        context_variables: Optional context variables for the research
    """
    try:
        researcher = GPTResearcher(query=query, report_type=report_type, config_path=None)
        await researcher.conduct_research()
        report = await researcher.write_report()
        return Result(value=report)
    except Exception as e:
        return Result(value=f"Error conducting research: {str(e)}")

def research_agent_instructions(context_variables: Optional[Dict[str, Any]] = None) -> str:
    """
    Dynamic instructions for the research agent.
    
    Args:
        context_variables: Optional context variables to customize instructions
    """
    report_types = context_variables.get("report_types", REPORT_TYPES) if context_variables else REPORT_TYPES
    
    return f"""You are a sophisticated research agent that helps users find information on any topic.
    
Your capabilities:
1. Conduct in-depth research on any topic using web search and analysis
2. Generate different types of reports ({', '.join(report_types)})
3. Maintain context across multiple research queries
4. Remember previous findings and build upon them

When interacting:
1. Ask clarifying questions if the research query is ambiguous
2. Suggest related topics that might be interesting to explore
3. Maintain conversation history to provide coherent follow-ups
4. Use the conduct_research function when a new research query is clear

Available report types:
- detailed_report: Comprehensive analysis with multiple sources
- brief_report: Concise summary of key findings
- bullet_points: Key points in bullet format"""

# Create the research agent
research_agent = Agent(
    name="Research Agent",
    model="gpt-4o",
    instructions=research_agent_instructions,
    functions=[conduct_research],
)

def run_research_agent():
    """
    Run an interactive session with the research agent.
    """
    client = tangent()
    
    # Run the interactive loop with streaming enabled for real-time responses
    run_tangent_loop(
        starting_agent=research_agent,
        context_variables={"report_types": REPORT_TYPES},
        stream=True,
        debug=True
    )

if __name__ == "__main__":
    run_research_agent()