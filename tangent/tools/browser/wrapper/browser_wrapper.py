"""
Wrapper for browser-use package to integrate with tangent.
This provides a simple interface for browser automation within tangent.
"""

import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from browser_use.agent.views import ActionResult
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller

class BrowserAction(BaseModel):
    """Represents a browser action with its parameters."""
    action_name: str
    parameters: dict

class BrowserState(BaseModel):
    """Represents the current state of the browser."""
    url: str
    tabs: List[str]
    interactive_elements: str

    @classmethod
    def from_browser_state(cls, state):
        """Create BrowserState from browser-use state object."""
        return cls(
            url=state.url,
            tabs=[tab.url for tab in state.tabs],
            interactive_elements=state.element_tree.clickable_elements_to_string()
        )

class BrowserWrapper:
    """
    Wrapper for browser-use package.
    Provides a simple interface for browser automation within tangent.
    """
    
    def __init__(self, headless: bool = False):
        """Initialize the browser wrapper."""
        self.controller = Controller()
        self.browser = Browser(config=BrowserConfig(headless=headless))
        self.browser_context: Optional[BrowserContext] = None
        self.ActionModel = self.controller.registry.create_action_model()

    async def __aenter__(self) -> 'BrowserWrapper':
        """Set up browser context when used as async context manager."""
        self.browser_context = await self.browser.new_context()
        await self.browser_context.create_new_tab("about:blank")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up browser resources."""
        if self.browser_context:
            await self.browser_context.close()
        if self.browser:
            await self.browser.close()

    async def get_state(self) -> BrowserState:
        """Get current browser state."""
        if not self.browser_context:
            raise ValueError("Browser context not initialized")
        state = await self.browser_context.get_state()
        return BrowserState.from_browser_state(state)

    async def execute_action(self, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a browser action.
        
        Args:
            action_name: Name of the action (e.g., "search_google", "go_to_url")
            parameters: Parameters for the action (e.g., {"query": "search term"})
            
        Returns:
            Dictionary containing action results and new browser state
        """
        if not self.browser_context:
            raise ValueError("Browser context not initialized")

        try:
            # Create action model
            action_model = self.ActionModel(**{action_name: parameters})
            
            # Execute action
            results = await self.controller.multi_act(
                [action_model],
                self.browser_context
            )
            
            # Get new state
            new_state = await self.get_state()
            
            return {
                "success": True,
                "action": {
                    "name": action_name,
                    "parameters": parameters
                },
                "state": new_state.model_dump(),
                "results": [str(result) for result in results]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "state": (await self.get_state()).model_dump()
            }

# Example usage
async def main():
    """Example of how to use the browser wrapper."""
    async with BrowserWrapper(headless=False) as browser:
        # Example: Search Google
        result = await browser.execute_action(
            "search_google",
            {"query": "Python programming"}
        )
        print(f"Action result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
