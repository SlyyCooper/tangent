from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Type, Dict, Union

from tangent.types import (
	MessageContent,
	_TextContent,
	_ImageContent,
	Response,
	Structured_Result
)

from tangent.browser.agent.message_manager.views import MessageHistory, MessageMetadata
from tangent.browser.agent.prompts import AgentMessagePrompt, SystemPrompt
from tangent.browser.agent.views import ActionResult, AgentOutput, AgentStepInfo
from tangent.browser.browser.views import BrowserState

logger = logging.getLogger(__name__)


class MessageManager:
	def __init__(
		self,
		task: str,
		action_descriptions: str,
		system_prompt_class: Type[SystemPrompt],
		max_input_tokens: int = 128000,
		estimated_tokens_per_character: float = 0.25,
		image_tokens: int = 129,  # Default for high-res images
		include_attributes: list[str] = [],
		max_error_length: int = 400,
		max_actions_per_step: int = 10,
		token_buffer: int = 1000,
	):
		self.system_prompt_class = system_prompt_class
		self.max_input_tokens = max_input_tokens
		self.token_buffer = token_buffer
		self.history = MessageHistory()
		self.task = task
		self.action_descriptions = action_descriptions
		self.ESTIMATED_TOKENS_PER_CHARACTER = estimated_tokens_per_character
		self.IMG_TOKENS = image_tokens
		self.include_attributes = include_attributes
		self.max_error_length = max_error_length

		# Add system message
		system_message = self.system_prompt_class(
			self.action_descriptions,
			current_date=datetime.now(),
			max_actions_per_step=max_actions_per_step,
		).get_system_message()
		self._add_message_with_tokens(system_message)
		
		# Add task message
		task_message = {
			"role": "user",
			"content": f"Your task is: {task}"
		}
		self._add_message_with_tokens(task_message)

	def add_state_message(
		self,
		state: BrowserState,
		result: Optional[List[ActionResult]] = None,
		step_info: Optional[AgentStepInfo] = None,
	) -> None:
		"""Add browser state and optional result to message history"""

		# if keep in memory, add directly to history and add state without result
		if result:
			for r in result:
				if r.include_in_memory:
					if r.result_overview:  # Use result_overview from Structured_Result
						msg = {
							"role": "user",
							"content": str(r.result_overview)
						}
						self._add_message_with_tokens(msg)
					if r.error:
						msg = {
							"role": "user",
							"content": str(r.error)[-self.max_error_length:]
						}
						self._add_message_with_tokens(msg)
					result = None  # if result in history, we dont want to add it again

		# Add state message
		state_message = AgentMessagePrompt(
			state,
			result,
			include_attributes=self.include_attributes,
			max_error_length=self.max_error_length,
			step_info=step_info,
		).get_user_message()
		self._add_message_with_tokens(state_message)

	def _remove_last_state_message(self) -> None:
		"""Remove last state message from history"""
		if len(self.history.messages) > 2 and self.history.messages[-1].message["role"] == "user":
			self.history.remove_message()

	def add_model_output(self, model_output: AgentOutput) -> None:
		"""Add model output as assistant message"""
		content = (
			f"Evaluation of previous goal: {model_output.current_state.evaluation_previous_goal}\n"
			f"Memory: {model_output.current_state.memory}\n"
			f"Next goal: {model_output.current_state.next_goal}\n"
			f"Action: {[action.model_dump() for action in model_output.action]}"
		)
		msg = {
			"role": "assistant",
			"content": content
		}
		self._add_message_with_tokens(msg)

	def get_messages(self) -> List[Dict[str, Union[str, MessageContent]]]:
		"""Get current message list, potentially trimmed to max tokens"""
		self.cut_messages()
		return [m.message for m in self.history.messages]

	def _add_message_with_tokens(self, message: Dict[str, Union[str, MessageContent]]) -> None:
		"""Add message with token count to history"""
		content = message["content"]
		tokens = 0
		
		if isinstance(content, list):
			for item in content:
				if item.get("type") == "image_url":
					tokens += self.IMG_TOKENS
				elif item.get("type") == "text":
					tokens += self._count_text_tokens(item["text"])
		else:
			tokens = self._count_text_tokens(content)
			
		metadata = MessageMetadata(input_tokens=tokens)
		self.history.add_message(message, metadata)

	def _count_text_tokens(self, text: str) -> int:
		"""Estimate token count for text"""
		return int(len(text) * self.ESTIMATED_TOKENS_PER_CHARACTER)

	def cut_messages(self) -> None:
		"""Trim message history to fit within token limit"""
		# Calculate available tokens considering buffer
		available_tokens = self.max_input_tokens - self.token_buffer
		diff = self.history.total_tokens - available_tokens
		
		if diff <= 0:
			return None

		# Keep track of essential messages (system and task)
		essential_messages = self.history.messages[:2]
		essential_tokens = sum(msg.metadata.input_tokens for msg in essential_messages)
		
		# Calculate remaining tokens for other messages
		remaining_tokens = available_tokens - essential_tokens
		
		if remaining_tokens <= 0:
			raise ValueError(
				'Token limit too low - cannot maintain system and task messages. '
				'Please increase max_input_tokens or reduce system prompt/task length.'
			)

		# Start with most recent messages and work backwards
		current_messages = []
		current_tokens = 0
		
		for msg in reversed(self.history.messages[2:]):
			msg_tokens = msg.metadata.input_tokens
			
			# If adding this message would exceed limit, try to trim it
			if current_tokens + msg_tokens > remaining_tokens:
				# Calculate how many tokens we can keep
				available_msg_tokens = remaining_tokens - current_tokens
				
				if available_msg_tokens > 100:  # Only keep message if we can keep meaningful content
					content = msg.message["content"]
					if isinstance(content, list):
						# Handle messages with images
						new_content = []
						new_tokens = 0
						for item in content:
							if isinstance(item, _ImageContent):
								if new_tokens + self.IMG_TOKENS <= available_msg_tokens:
									new_content.append(item)
									new_tokens += self.IMG_TOKENS
							elif isinstance(item, _TextContent):
								text_tokens = self._count_text_tokens(item.text)
								if new_tokens + text_tokens <= available_msg_tokens:
									new_content.append(item)
									new_tokens += text_tokens
						if new_content:
							msg.message["content"] = new_content
							msg.metadata.input_tokens = new_tokens
							current_messages.append(msg)
							current_tokens += new_tokens
					else:
						# Handle text messages
						token_ratio = available_msg_tokens / msg_tokens
						chars_to_keep = int(len(content) * token_ratio)
						if chars_to_keep > 50:  # Only keep if we have meaningful content
							trimmed_content = content[:chars_to_keep]
							new_tokens = self._count_text_tokens(trimmed_content)
							msg.message["content"] = trimmed_content
							msg.metadata.input_tokens = new_tokens
							current_messages.append(msg)
							current_tokens += new_tokens
				break
			else:
				current_messages.append(msg)
				current_tokens += msg_tokens
				
		self.history.messages = essential_messages + list(reversed(current_messages))
		self.history.total_tokens = essential_tokens + current_tokens
		
		logger.debug(
			f'Memory trimmed to {self.history.total_tokens}/{self.max_input_tokens} tokens '
			f'({len(self.history.messages)} messages)'
		)
