from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import textwrap
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, List, Dict, Union
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from tangent import Agent as TangentAgent, Response, Structured_Result
from tangent.types import (
	MessageContent,
	_TextContent,
	_ImageContent,
	RateLimitError
)
from tangent.core import tangent

from tangent.browser.agent.message_manager.service import MessageManager
from tangent.browser.agent.prompts import AgentMessagePrompt, SystemPrompt
from tangent.browser.agent.views import (
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentStepInfo,
)
from tangent.browser.browser.browser import Browser
from tangent.browser.browser.context import BrowserContext
from tangent.browser.browser.views import BrowserState, BrowserStateHistory
from tangent.browser.controller.registry.views import ActionModel
from tangent.browser.controller.service import Controller
from tangent.browser.dom.history_tree_processor.service import (
	DOMHistoryElement,
	HistoryTreeProcessor,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class BrowserAgent:
	def __init__(
		self,
		task: str,
		tangent_agent: TangentAgent,
		browser: Browser | None = None,
		browser_context: BrowserContext | None = None,
		controller: Controller = Controller(),
		use_vision: bool = True,
		save_conversation_path: Optional[str] = None,
		max_failures: int = 5,
		retry_delay: int = 10,
		system_prompt_class: Type[SystemPrompt] = SystemPrompt,
		validate_output: bool = False,
		generate_gif: bool = True,
		include_attributes: list[str] = [
			'title',
			'type',
			'name',
			'role',
			'tabindex',
			'aria-label',
			'placeholder',
			'value',
			'alt',
			'aria-expanded',
		],
		max_error_length: int = 400,
		max_actions_per_step: int = 10,
	):
		self.agent_id = str(uuid.uuid4())
		self.task = task
		self.use_vision = use_vision
		self.tangent_agent = tangent_agent
		self.save_conversation_path = save_conversation_path
		self._last_result = None
		self.include_attributes = include_attributes
		self.max_error_length = max_error_length
		self.generate_gif = generate_gif
		
		# Initialize tangent client
		self.client = tangent()
		
		# Controller setup
		self.controller = controller
		self.max_actions_per_step = max_actions_per_step

		# Browser setup
		self.injected_browser = browser is not None
		self.injected_browser_context = browser_context is not None

		# Initialize browser first if needed
		self.browser = browser if browser is not None else (None if browser_context else Browser())

		# Initialize browser context
		if browser_context:
			self.browser_context = browser_context
		elif self.browser:
			self.browser_context = BrowserContext(
				browser=self.browser, config=self.browser.config.new_context_config
			)
		else:
			self.browser = Browser()
			self.browser_context = BrowserContext(browser=self.browser)

		self.system_prompt_class = system_prompt_class

		# Action and output models setup
		self._setup_action_models()
		
		# Message history setup
		self.messages = []
		self.extracted_data = {}
		
		# Add system message
		system_prompt = self.system_prompt_class(
			self.controller.registry.get_prompt_description(),
			current_date=datetime.now(),
			max_actions_per_step=max_actions_per_step,
		).get_system_message()
		self.messages.append({
			"role": "system",
			"content": system_prompt
		})
		
		# Add task message
		task_message = {
			"role": "user",
			"content": str(f"Your task is: {task}")
		}
		self.messages.append(task_message)
		
		# Tracking variables
		self.history: AgentHistoryList = AgentHistoryList(history=[])
		self.n_steps = 1
		self.consecutive_failures = 0
		self.max_failures = max_failures
		self.retry_delay = retry_delay
		self.validate_output = validate_output

		if save_conversation_path:
			logger.info(f'Saving conversation to {save_conversation_path}')

	def _setup_action_models(self) -> None:
		"""Setup dynamic action models from controller's registry"""
		# Get the dynamic action model from controller's registry
		self.ActionModel = self.controller.registry.create_action_model()
		# Create output model with the dynamic actions
		self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

	async def step(self) -> None:
		"""Execute one step of the agent's task"""
		state = None
		results = None
		model_output = None

		try:
			# Get current browser state
			state = await self.browser_context.get_state()
			
			# Create state message
			state_message = AgentMessagePrompt(
				state,
				self._last_result,
				include_attributes=self.include_attributes,
				max_error_length=self.max_error_length,
				step_info=AgentStepInfo(self.n_steps, self.max_actions_per_step),
			).get_user_message()
			
			# Add state message to history
			self.messages.append({
				"role": "user",
				"content": json.dumps(state_message) if isinstance(state_message, dict) else state_message
			})
			
			# Get response from tangent
			response = self.client.run(
				agent=self.tangent_agent,
				messages=self.messages,
				extracted_data=self.extracted_data
			)
			
			# Update extracted data
			self.extracted_data.update(response.extracted_data)
			
			# Parse model output
			model_output = self.AgentOutput.model_validate_json(response.messages[-1].content)
			
			# Execute actions
			results = await self._execute_actions(model_output.action)
			self._last_result = results
			
			# Validate output if enabled
			if self.validate_output:
				await self._validate_output(state, model_output, results)
			
			# Reset failure counter on success
			self.consecutive_failures = 0
			
		except Exception as e:
			results = self._handle_step_error(e)
			self._last_result = results

		finally:
			if state and results:
				self._make_history_item(model_output, state, results)

	def _handle_step_error(self, error: Exception) -> List[ActionResult]:
		"""Handle errors during step execution"""
		error_msg = AgentError.format_error(error)
		logger.error(f'Step error: {error_msg}')
		
		# Handle rate limit errors by waiting
		if isinstance(error, RateLimitError):
			time.sleep(self.retry_delay)
			self.consecutive_failures = 0  # Reset on rate limit
		else:
			self.consecutive_failures += 1
			
		# If too many failures, create a final error result
		if self.consecutive_failures >= self.max_failures:
			return [ActionResult(
				result_overview=f"Failed after {self.max_failures} attempts: {error_msg}",
				error=error_msg,
				is_done=True
			)]
			
		# Otherwise create a regular error result
		return [ActionResult(
			result_overview=f"Error during step {self.n_steps}: {error_msg}",
			error=error_msg
		)]

	async def _execute_actions(self, actions: List[ActionModel]) -> List[ActionResult]:
		"""Execute a list of actions and return their results"""
		results = []
		for action in actions[:self.max_actions_per_step]:
			try:
				result = await self.controller.execute(action, self.browser_context)
				results.append(result)
				if result.is_done:
					break
			except Exception as e:
				error_msg = str(e)[-self.max_error_length:]
				results.append(ActionResult(
					result_overview=f"Action failed: {error_msg}",
					error=error_msg
				))
				break
		return results

	async def _validate_output(
		self,
		state: BrowserState,
		model_output: AgentOutput,
		results: List[ActionResult]
	) -> None:
		"""Validate the output using a validation agent"""
		if not self.browser_context:
			return
			
		# Create validation agent
		validation_agent = TangentAgent(
			name="Validator",
			model=self.tangent_agent.model,
			instructions="You are a validator that checks if the browser agent's actions were successful."
		)
		
		# Create validation message
		validation_message = AgentMessagePrompt(
			state,
			results,
			include_attributes=self.include_attributes,
			max_error_length=self.max_error_length,
		).get_user_message()
		
		# Get validation response
		response = self.client.run(
			agent=validation_agent,
			messages=[validation_message],
			extracted_data=self.extracted_data
		)
		
		# Log validation result
		logger.info(f'Validation result: {response.messages[-1].content}')

	def _make_history_item(
		self,
		model_output: Optional[AgentOutput],
		state: BrowserState,
		results: List[ActionResult]
	) -> None:
		"""Create and save a history item"""
		state_history = BrowserStateHistory(
			url=state.url,
			title=state.title,
			screenshot=state.screenshot,
			tabs=state.tabs if hasattr(state, 'tabs') else [],
			interacted_element=None if not model_output else AgentHistory.get_interacted_element(
				model_output,
				state.selector_map
			)
		)
		
		history_item = AgentHistory(
			model_output=model_output,
			result=results,
			state=state_history
		)
		
		self.history.history.append(history_item)
		
		if self.save_conversation_path:
			self._save_conversation()

	def _save_conversation(self) -> None:
		"""Save the conversation history to a file"""
		try:
			self.history.save_to_file(self.save_conversation_path)
		except Exception as e:
			logger.error(f'Failed to save conversation: {e}')

