"""
Tangent integration layer for browser automation.
This integrates our browser wrapper into the tangent package.
"""

from typing import Optional, Dict, Any, Tuple, List, Literal
from tangent import setup_agent, Structured_Result
from wrapper import BrowserWrapper  # Fixed import path

class BrowserAgent:
    """
    Browser-enabled agent for tangent.
    Provides browser automation capabilities through tangent's interface.
    """
    
    def __init__(
        self,
        name: str = "BrowserBot",
        model: str = "gpt-4o",
        headless: bool = False
    ):
        """Initialize the browser-enabled agent."""
        # Set up tangent agent
        self.client, self.agent = setup_agent(
            name=name,
            model=model,
            instructions="""You are a browser automation expert that helps users navigate and interact with web pages.
            You can perform the following actions:
            - search_google: Search Google (parameters: query)
            - go_to_url: Navigate to a specific URL
            - go_back: Go back to the previous page
            - click_element: Click an element by index
            - input_text: Input text into a field by index
            - scroll_up: Scroll page up (optional amount)
            - scroll_down: Scroll page down (optional amount)
            - open_tab: Open a new tab with optional URL
            - switch_tab: Switch to a specific tab by index
            - extract_content: Extract page content as text/markdown/html
            - get_dropdown_options: Get options from a dropdown menu
            - select_dropdown_option: Select an option from a dropdown
            
            Always respond with clear, structured actions and maintain state awareness.
            """
        )
        
        # Initialize browser wrapper
        self.browser = BrowserWrapper(headless=headless)
        self._browser_context = None
    
    async def __aenter__(self) -> 'BrowserAgent':
        """Set up browser context when used as async context manager."""
        self._browser_context = await self.browser.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up browser resources."""
        await self.browser.__aexit__(exc_type, exc_val, exc_tb)

    def _format_browser_result(self, result: Dict[str, Any]) -> Structured_Result:
        """Convert browser result to tangent's Structured_Result."""
        if result["success"]:
            return Structured_Result(
                result_overview=f"Successfully executed {result['action']['name']}",
                extracted_data={
                    "browser_state": result["state"],
                    "action_results": result["results"]
                }
            )
        else:
            return Structured_Result(
                result_overview=f"Error: {result.get('error', 'Unknown error')}",
                extracted_data={
                    "browser_state": result["state"]
                }
            )

    async def search_google(self, query: str) -> Structured_Result:
        """
        Search Google.
        
        Args:
            query: Search query
            
        Returns:
            Structured_Result containing search results and browser state
        """
        result = await self.browser.execute_action(
            "search_google",
            {"query": query}
        )
        return self._format_browser_result(result)

    async def go_to_url(self, url: str) -> Structured_Result:
        """
        Navigate to a specific URL.
        
        Args:
            url: The URL to navigate to
            
        Returns:
            Structured_Result containing navigation result and browser state
        """
        result = await self.browser.execute_action(
            "go_to_url",
            {"url": url}
        )
        return self._format_browser_result(result)

    async def go_back(self) -> Structured_Result:
        """
        Go back to the previous page.
        
        Returns:
            Structured_Result containing navigation result and browser state
        """
        result = await self.browser.execute_action(
            "go_back",
            {}  # No parameters needed
        )
        return self._format_browser_result(result)

    async def click_element(self, index: int) -> Structured_Result:
        """
        Click an element on the page by its index.
        
        Args:
            index: The index of the element to click (1-based)
            
        Returns:
            Structured_Result containing click result and browser state
        """
        result = await self.browser.execute_action(
            "click_element",
            {"index": index}
        )
        return self._format_browser_result(result)

    async def input_text(self, index: int, text: str) -> Structured_Result:
        """
        Input text into a field by its index.
        
        Args:
            index: The index of the input field (1-based)
            text: The text to input
            
        Returns:
            Structured_Result containing input result and browser state
        """
        result = await self.browser.execute_action(
            "input_text",
            {
                "index": index,
                "text": text
            }
        )
        return self._format_browser_result(result)

    async def scroll_up(self, amount: Optional[int] = None) -> Structured_Result:
        """
        Scroll the page up.
        
        Args:
            amount: Optional amount to scroll (pixels). If None, scrolls a default amount.
            
        Returns:
            Structured_Result containing scroll result and browser state
        """
        params = {"direction": "up"}
        if amount is not None:
            params["amount"] = amount
            
        result = await self.browser.execute_action(
            "scroll",
            params
        )
        return self._format_browser_result(result)

    async def scroll_down(self, amount: Optional[int] = None) -> Structured_Result:
        """
        Scroll the page down.
        
        Args:
            amount: Optional amount to scroll (pixels). If None, scrolls a default amount.
            
        Returns:
            Structured_Result containing scroll result and browser state
        """
        params = {"direction": "down"}
        if amount is not None:
            params["amount"] = amount
            
        result = await self.browser.execute_action(
            "scroll",
            params
        )
        return self._format_browser_result(result)

    async def open_tab(self, url: Optional[str] = None) -> Structured_Result:
        """
        Open a new browser tab.
        
        Args:
            url: Optional URL to navigate to in the new tab
            
        Returns:
            Structured_Result containing tab creation result and browser state
        """
        # First create the tab
        result = await self.browser.execute_action(
            "open_tab",
            {"url": "about:blank"}
        )
        
        # If URL provided, navigate to it
        if url and result["success"]:
            nav_result = await self.browser.execute_action(
                "go_to_url",
                {"url": url}
            )
            # Merge results
            result["results"].extend(nav_result.get("results", []))
            result["state"] = nav_result["state"]  # Use final state after navigation
            
        return self._format_browser_result(result)

    async def switch_tab(self, tab_index: int) -> Structured_Result:
        """
        Switch to a specific browser tab.
        
        Args:
            tab_index: The index of the tab to switch to (0-based)
            
        Returns:
            Structured_Result containing tab switch result and browser state
        """
        result = await self.browser.execute_action(
            "switch_tab",
            {"page_id": tab_index}  # browser-use expects page_id
        )
        return self._format_browser_result(result)

    async def extract_content(self, format: Literal['text', 'markdown', 'html'] = 'text') -> Structured_Result:
        """
        Extract content from the current page.
        
        Args:
            format: The format to extract content in ('text', 'markdown', or 'html')
            
        Returns:
            Structured_Result containing extracted content and browser state
        """
        result = await self.browser.execute_action(
            "extract_content",
            {"value": format}
        )
        return self._format_browser_result(result)

    async def get_dropdown_options(self, index: int) -> Structured_Result:
        """
        Get options from a dropdown menu.
        
        Args:
            index: The index of the dropdown element (1-based)
            
        Returns:
            Structured_Result containing dropdown options and browser state
        """
        result = await self.browser.execute_action(
            "get_dropdown_options",
            {"index": index}
        )
        return self._format_browser_result(result)

    async def select_dropdown_option(self, index: int, text: str) -> Structured_Result:
        """
        Select an option from a dropdown menu.
        
        Args:
            index: The index of the dropdown element (1-based)
            text: The text of the option to select
            
        Returns:
            Structured_Result containing selection result and browser state
        """
        result = await self.browser.execute_action(
            "select_dropdown_option",
            {
                "index": index,
                "text": text
            }
        )
        return self._format_browser_result(result)

# Example usage
async def main():
    """Example of how to use the browser agent."""
    async with BrowserAgent(headless=False) as agent:
        # Example: Search Google
        result = await agent.search_google("Python programming")
        print(f"Search result: {result.result_overview}")
        print(f"Browser state: {result.extracted_data['browser_state']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 