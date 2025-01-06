from __future__ import annotations

from typing import List, Dict, Union, Optional

from pydantic import BaseModel, Field, ConfigDict
from tangent.types import MessageContent, _TextContent, _ImageContent


class MessageMetadata(BaseModel):
	"""Metadata for a message including token counts"""
	model_config = ConfigDict(extra="allow")

	input_tokens: int = 0


class ManagedMessage(BaseModel):
	"""A message with its metadata
	
	Attributes:
		message: The message in standard format (role, content)
		metadata: Token count and other metadata
	"""
	model_config = ConfigDict(extra="allow")

	message: Dict[str, Union[str, MessageContent]]
	metadata: MessageMetadata = Field(default_factory=MessageMetadata)


class MessageHistory(BaseModel):
	"""Container for message history with metadata
	
	Attributes:
		messages: List of messages with their metadata
		total_tokens: Running total of tokens used
	"""
	model_config = ConfigDict(extra="allow")

	messages: List[ManagedMessage] = Field(default_factory=list)
	total_tokens: int = 0

	def add_message(
		self, 
		message: Dict[str, Union[str, MessageContent]], 
		metadata: Optional[MessageMetadata] = None
	) -> None:
		"""Add a message with metadata to history
		
		Args:
			message: Message in standard format
			metadata: Optional metadata (creates empty if None)
		"""
		if metadata is None:
			metadata = MessageMetadata()
			
		self.messages.append(ManagedMessage(message=message, metadata=metadata))
		self.total_tokens += metadata.input_tokens

	def remove_message(self, index: int = -1) -> None:
		"""Remove a message from history
		
		Args:
			index: Index of message to remove (default: last message)
		"""
		if self.messages:
			msg = self.messages.pop(index)
			self.total_tokens -= msg.metadata.input_tokens

	def get_messages(self) -> List[Dict[str, Union[str, MessageContent]]]:
		"""Get all messages in standard format"""
		return [msg.message for msg in self.messages]

	def clear(self) -> None:
		"""Clear all messages and reset token count"""
		self.messages.clear()
		self.total_tokens = 0
