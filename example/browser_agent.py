import asyncio
from pathlib import Path
from typing import List, Optional

from browser_use import Agent as BrowserAgent, Browser, BrowserConfig, SystemPrompt
from langchain_openai import ChatOpenAI

class CustomSystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        existing_rules = super().important_rules()
        custom_rules = """
9. BROWSER INTERACTION:
   - Always analyze the current page state before taking actions
   - Use vision capabilities to understand page layout and element relationships
   - Handle errors gracefully and try alternative approaches
   - Chain multiple actions when appropriate (e.g. form filling)
   - Use scroll actions to find elements not currently visible
   - Handle popups and cookie banners appropriately
   - Extract content when needed to verify task completion
   - Use done action when task is complete
"""
        return f'{existing_rules}\n{custom_rules}'

async def main():
    # Example task
    task = "Browse websites and help users find information"
    
    # Initialize the LangChain model
    llm = ChatOpenAI(model="gpt-4o") 
    
    # Initialize browser config
    browser_config = BrowserConfig(
        headless=False,
        disable_security=True,
        extra_chromium_args=["--disable-web-security"]
    )
    
    # Initialize browser
    browser = Browser(config=browser_config)
    
    # Create the browser agent
    agent = BrowserAgent(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=True,
        save_conversation_path="./chat_history/conversation.json",
        system_prompt_class=CustomSystemPrompt,
        validate_output=True,
        generate_gif=True
    )
    
    try:
        # Example conversation
        prompts = [
            "Go to huggingface.co",
            "What models are trending?",
            "Click on the first trending model",
            "What are the model details?",
            "Go back to the main page",
            "Search for 'stable diffusion'",
        ]
        
        for prompt in prompts:
            print(f"\nUser: {prompt}")
            
            # Update task and run step
            agent.task = prompt
            await agent.step()
            
            # Get latest history
            if agent.history.history:
                latest = agent.history.history[-1]
                
                # Print results
                print(f"Visited URL: {latest.state.url}")
                if latest.model_output:
                    print(f"Actions taken: {[a.model_dump() for a in latest.model_output.action]}")
                if latest.result:
                    for result in latest.result:
                        if result.extracted_content:
                            print(f"Extracted content: {result.extracted_content}")
                        if result.error:
                            print(f"Error: {result.error}")
                        if result.is_done:
                            print("Task completed!")
            
            await asyncio.sleep(2)  # Add a small delay between actions
            
    finally:
        # Clean up
        if not agent.injected_browser:
            await browser.close()

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
