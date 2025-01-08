from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from tangent.types import MessageContent


class MessageMetadata(BaseModel):
	"""Metadata for a message including token counts"""

	input_tokens: int = 0


class ManagedMessage(BaseModel):
	"""A message with its metadata"""

	message: MessageContent
	metadata: MessageMetadata = Field(default_factory=MessageMetadata)


class MessageHistory(BaseModel):
	"""Container for message history with metadata"""

	messages: List[ManagedMessage] = Field(default_factory=list)
	total_tokens: int = 0

	def add_message(self, message: MessageContent, metadata: MessageMetadata) -> None:
		"""Add a message with metadata"""
		self.messages.append(ManagedMessage(message=message, metadata=metadata))
		self.total_tokens += metadata.input_tokens

	def remove_message(self, index: int = -1) -> None:
		"""Remove last message from history"""
		if self.messages:
			msg = self.messages.pop(index)
			self.total_tokens -= msg.metadata.input_tokens