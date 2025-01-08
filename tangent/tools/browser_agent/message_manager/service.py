from __future__ import annotations

import logging
from typing import List, Optional, Union

from pydantic import BaseModel

from ..prompts import AgentMessagePrompt, SystemPrompt
from tangent import tangent, Response
from tangent.types import MessageContent
from tangent.util import debug_print

logger = logging.getLogger(__name__)


class MessageMetadata(BaseModel):
	"""Metadata for message tracking and token management.
	
	Attributes:
		input_tokens: Number of tokens in the message
		conversation_id: Optional ID for conversation tracking
	"""
	input_tokens: int = 0
	conversation_id: Optional[str] = None


class AgentHistory(BaseModel):
	message: MessageContent
	metadata: MessageMetadata


class AgentHistoryList(BaseModel):
	history: List[AgentHistory]
	total_tokens: int = 0
	conversation_id: Optional[str] = None

	def add_message(self, message: MessageContent, metadata: MessageMetadata) -> None:
		self.history.append(AgentHistory(message=message, metadata=metadata))
		self.total_tokens += metadata.input_tokens
		# Update conversation_id if not set
		if not self.conversation_id and metadata.conversation_id:
			self.conversation_id = metadata.conversation_id

	def remove_message(self, index: int) -> None:
		if not self.history:
			return
		if index < 0:
			index = len(self.history) + index
		if index < 0 or index >= len(self.history):
			return
		removed_message = self.history.pop(index)
		self.total_tokens -= removed_message.metadata.input_tokens

	def clear(self) -> None:
		self.history = []
		self.total_tokens = 0
		# Keep conversation_id for persistence


class MessageManager:
	IMG_TOKENS = 1000
	ESTIMATED_TOKENS_PER_CHARACTER = 4
	def __init__(
		self,
		llm: tangent,
		task: str,
		action_descriptions: str,
		system_prompt_class: type[SystemPrompt] = SystemPrompt,
		max_input_tokens: int = 128000,
		include_attributes: list[str] = [],
		max_error_length: int = 400,
		max_actions_per_step: int = 10,
		tool_call_in_content: bool = True,
		debug: bool = False,
		conversation_id: Optional[str] = None,
	):
		self.llm = llm
		self.task = task
		self.action_descriptions = action_descriptions
		self.system_prompt = system_prompt_class(
			task=self.task,
			action_descriptions=self.action_descriptions,
			include_attributes=include_attributes,
			max_error_length=max_error_length,
			max_actions_per_step=max_actions_per_step,
			tool_call_in_content=tool_call_in_content,
		)
		self.max_input_tokens = max_input_tokens
		self.history = AgentHistoryList(history=[], conversation_id=conversation_id)
		self.debug = debug

	def clear_messages(self) -> None:
		"""Clear messages but maintain conversation ID"""
		conversation_id = self.history.conversation_id
		self.history.clear()
		self.history.conversation_id = conversation_id

	def get_messages(self) -> list[dict]:
		"""Get current message list with conversation context"""
		messages = []
		for item in self.history.history:
			messages.append({
				"role": "user",
				"content": item.message,
				"conversation_id": item.metadata.conversation_id or self.history.conversation_id
			})
		return messages

	def add_state_message(
		self, state: dict, last_result: Optional[list[dict]], step_info: Optional[dict] = None
	) -> None:
		"""Add state message to history with conversation tracking"""
		prompt = self.system_prompt.get_state_prompt(state, last_result, step_info)
		msg = {
			"role": "user",
			"content": prompt,
			"conversation_id": self.history.conversation_id
		}
		self._add_message_with_tokens(msg)

	def add_model_output(self, model_output: dict) -> None:
		"""Add model output to history with conversation tracking"""
		msg = {
			"role": "assistant",
			"content": model_output,
			"conversation_id": self.history.conversation_id
		}
		self._add_message_with_tokens(msg)

	async def load_conversation(self, conversation_id: str) -> None:
		"""Load a conversation from Tangent's storage"""
		try:
			# Use Tangent's conversation loading
			response: Response = await self.llm.load_conversation(conversation_id)
			if response and response.messages:
				self.clear_messages()
				self.history.conversation_id = conversation_id
				for msg in response.messages:
					self._add_message_with_tokens(msg)
		except Exception as e:
			logger.error(f"Failed to load conversation {conversation_id}: {e}")

	async def save_conversation(self) -> None:
		"""Save the current conversation to Tangent's storage"""
		if not self.history.conversation_id:
			debug_print(self.debug, "No conversation ID to save")
			return
		
		try:
			messages = self.get_messages()
			debug_print(self.debug, f"Saving conversation {self.history.conversation_id} with {len(messages)} messages")
			await self.llm.save_conversation(
				conversation_id=self.history.conversation_id,
				messages=messages
			)
		except Exception as e:
			logger.error(f"Failed to save conversation {self.history.conversation_id}: {e}")

	def _add_message_with_tokens(self, message: dict) -> None:
		"""Add message with token count metadata and conversation tracking"""
		token_count = self._count_tokens(message)
		metadata = MessageMetadata(
			input_tokens=token_count,
			conversation_id=message.get("conversation_id", self.history.conversation_id)
		)
		self.history.add_message(message["content"], metadata)

	def _count_tokens(self, message: dict) -> int:
		"""Count tokens in a message using the model's tokenizer"""
		tokens = 0
		if isinstance(message['content'], list):
			for item in message['content']:
				if 'image_url' in item:
					tokens += self.IMG_TOKENS
				elif isinstance(item, dict) and 'text' in item:
					tokens += self._count_text_tokens(item['text'])
		else:
			tokens += self._count_text_tokens(message['content'])
		return tokens

	def _count_text_tokens(self, text: str) -> int:
		"""Count tokens in a text string"""
		if isinstance(self.llm, tangent):
			try:
				tokens = self.llm.get_num_tokens(text)
			except Exception:
				tokens = (
					len(text) // self.ESTIMATED_TOKENS_PER_CHARACTER
				)  # Rough estimate if no tokenizer available
		else:
			tokens = (
				len(text) // self.ESTIMATED_TOKENS_PER_CHARACTER
			)  # Rough estimate if no tokenizer availa ble
		return tokens

	def _remove_last_state_message(self) -> None:
		"""Remove the last state message from history"""
		if not self.history.history:
			return
		if len(self.history.history) > 0:
			self.history.remove_message(index=-1)

	def cut_messages(self):
		"""Get current message list, potentially trimmed to max tokens"""
		diff = self.history.total_tokens - self.max_input_tokens
		if diff <= 0:
			return None

		msg = self.history.history[-1]

		# if list with image remove image
		if isinstance(msg.message, list):
			text = ''
			for item in msg.message:
				if 'image_url' in item:
					msg.message.remove(item)
					diff -= self.IMG_TOKENS
					msg.metadata.input_tokens -= self.IMG_TOKENS
					self.history.total_tokens -= self.IMG_TOKENS
					logger.debug(
						f'Removed image with {self.IMG_TOKENS} tokens - total tokens now: {self.history.total_tokens}/{self.max_input_tokens}'
					)
				elif 'text' in item and isinstance(item, dict):
					text += item['text']
			msg.message = text
			self.history.history[-1] = msg

		if diff <= 0:
			return None

		# if still over, remove text from state message proportionally to the number of tokens needed with buffer
		# Calculate the proportion of content to remove
		proportion_to_remove = diff / msg.metadata.input_tokens
		if proportion_to_remove > 0.99:
			raise ValueError(
				f'Max token limit reached - history is too long - reduce the system prompt or task less tasks or remove old messages. '
				f'proportion_to_remove: {proportion_to_remove}'
			)
		logger.debug(
			f'Removing {proportion_to_remove * 100:.2f}% of the last message  {proportion_to_remove * msg.metadata.input_tokens:.2f} / {msg.metadata.input_tokens:.2f} tokens)'
		)

		content = msg.message
		characters_to_remove = int(len(content) * proportion_to_remove)
		content = content[:-characters_to_remove]

		# remove tokens and old long message
		self.history.remove_message(index=-1)

		# new message with updated content
		msg = {"role": "user", "content": content}
		self._add_message_with_tokens(msg)

		last_msg = self.history.history[-1]

		logger.debug(
			f'Added message with {last_msg.metadata.input_tokens} tokens - total tokens now: {self.history.total_tokens}/{self.max_input_tokens} - total messages: {len(self.history.history)}'
		)