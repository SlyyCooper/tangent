from tangent.browser.logging_config import setup_logging

setup_logging()

from tangent.browser.agent.prompts import SystemPrompt as SystemPrompt
from tangent.browser.agent.service import BrowserAgent as Agent
from tangent.browser.agent.views import ActionModel as ActionModel
from tangent.browser.agent.views import ActionResult as ActionResult
from tangent.browser.agent.views import AgentHistoryList as AgentHistoryList
from tangent.browser.browser.browser import Browser as Browser
from tangent.browser.browser.browser import BrowserConfig as BrowserConfig
from tangent.browser.controller.service import Controller as Controller
from tangent.browser.dom.service import DomService as DomService

__all__ = [
	'Agent',
	'Browser',
	'BrowserConfig',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
]
