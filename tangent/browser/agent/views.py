from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, List, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from tangent.types import (
	Agent as TangentAgent,
	Response,
	Structured_Result,
	RateLimitError,
	MessageContent,
	_TextContent,
	_ImageContent
)

from tangent.browser.browser.views import BrowserStateHistory
from tangent.browser.controller.registry.views import ActionModel
from tangent.browser.dom.history_tree_processor.service import (
	DOMElementNode,
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from tangent.browser.dom.views import SelectorMap


@dataclass
class AgentStepInfo:
	"""Information about the current step in the agent's execution"""
	step_number: int
	max_steps: int


class ActionResult(Structured_Result):
	"""Result of executing an action, extends tangent's Structured_Result"""
	model_config = ConfigDict(extra="allow")

	is_done: bool = False
	include_in_memory: bool = False  # whether to include in past messages as context or not


class AgentBrain(BaseModel):
	"""Current state of the agent's reasoning"""
	model_config = ConfigDict(extra="allow")

	evaluation_previous_goal: str
	memory: str
	next_goal: str


class AgentOutput(BaseModel):
	"""Output model for agent responses
	
	This model is extended with custom actions in AgentService.
	Additional fields can be added through the DynamicActions model.
	"""
	model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

	current_state: AgentBrain
	action: List[ActionModel]

	@staticmethod
	def type_with_custom_actions(custom_actions: Type[ActionModel]) -> Type[AgentOutput]:
		"""Create a new AgentOutput type with custom action types"""
		return create_model(
			'AgentOutput',
			__base__=AgentOutput,
			action=(List[custom_actions], Field(...)),
			__module__=AgentOutput.__module__,
		)


class AgentHistory(BaseModel):
	"""History item for agent actions and their results"""
	model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=(), extra="allow")

	model_output: Optional[AgentOutput] = None
	result: List[ActionResult]
	state: BrowserStateHistory

	@staticmethod
	def get_interacted_element(
		model_output: AgentOutput, 
		selector_map: SelectorMap
	) -> List[Optional[DOMHistoryElement]]:
		"""Get the DOM elements that were interacted with"""
		elements: List[Optional[DOMHistoryElement]] = []
		for action in model_output.action:
			index = action.get_index()
			if index and index in selector_map:
				el: DOMElementNode = selector_map[index]
				elements.append(HistoryTreeProcessor.convert_dom_element_to_history_element(el))
			else:
				elements.append(None)
		return elements

	def model_dump(self, **kwargs) -> Dict[str, Any]:
		"""Custom serialization handling circular references"""
		# Handle action serialization
		model_output_dump = None
		if self.model_output:
			action_dump = [
				action.model_dump(exclude_none=True) 
				for action in self.model_output.action
			]
			model_output_dump = {
				'current_state': self.model_output.current_state.model_dump(),
				'action': action_dump,
			}

		return {
			'model_output': model_output_dump,
			'result': [r.model_dump(exclude_none=True) for r in self.result],
			'state': self.state.to_dict(),
		}


class AgentHistoryList(BaseModel):
	"""Container for a sequence of agent history items"""
	model_config = ConfigDict(extra="allow")

	history: List[AgentHistory]

	def __str__(self) -> str:
		return f'AgentHistoryList(all_results={self.action_results()}, all_model_outputs={self.model_actions()})'

	def save_to_file(self, filepath: Union[str, Path]) -> None:
		"""Save history to JSON file with proper serialization"""
		try:
			Path(filepath).parent.mkdir(parents=True, exist_ok=True)
			data = self.model_dump()
			with open(filepath, 'w', encoding='utf-8') as f:
				json.dump(data, f, indent=2)
		except Exception as e:
			raise ValueError(f"Failed to save history to {filepath}: {str(e)}") from e

	def model_dump(self, **kwargs) -> Dict[str, Any]:
		"""Custom serialization using AgentHistory's model_dump"""
		return {
			'history': [h.model_dump(**kwargs) for h in self.history],
		}

	@classmethod
	def load_from_file(
		cls, 
		filepath: Union[str, Path], 
		output_model: Type[AgentOutput]
	) -> AgentHistoryList:
		"""Load history from JSON file and validate with output model"""
		try:
			with open(filepath, 'r', encoding='utf-8') as f:
				data = json.load(f)
			
			# Validate and enrich history with custom actions
			for h in data['history']:
				if h['model_output']:
					if isinstance(h['model_output'], dict):
						h['model_output'] = output_model.model_validate(h['model_output'])
					else:
						h['model_output'] = None
				if 'interacted_element' not in h['state']:
					h['state']['interacted_element'] = None
					
			return cls.model_validate(data)
		except Exception as e:
			raise ValueError(f"Failed to load history from {filepath}: {str(e)}") from e

	def last_action(self) -> Optional[Dict[str, Any]]:
		"""Get the last action executed, if any"""
		if self.history and self.history[-1].model_output:
			return self.history[-1].model_output.action[-1].model_dump(exclude_none=True)
		return None

	def errors(self) -> List[str]:
		"""Get all error messages from history"""
		return [r.error for h in self.history for r in h.result if r.error]

	def final_result(self) -> Optional[str]:
		"""Get the final extracted content, if any"""
		if self.history and self.history[-1].result[-1].extracted_content:
			return self.history[-1].result[-1].extracted_content
		return None

	def is_done(self) -> bool:
		"""Check if the agent has completed its task"""
		return bool(
			self.history 
			and self.history[-1].result 
			and self.history[-1].result[-1].is_done
		)

	def has_errors(self) -> bool:
		"""Check if there were any errors during execution"""
		return bool(self.errors())

	def urls(self) -> List[str]:
		"""Get all unique URLs visited"""
		return [h.state.url for h in self.history if h.state.url]

	def screenshots(self) -> List[str]:
		"""Get all screenshots taken"""
		return [h.state.screenshot for h in self.history if h.state.screenshot]

	def action_names(self) -> List[str]:
		"""Get names of all actions executed"""
		return [list(action.keys())[0] for action in self.model_actions()]

	def model_thoughts(self) -> List[AgentBrain]:
		"""Get all agent reasoning states"""
		return [h.model_output.current_state for h in self.history if h.model_output]

	def model_outputs(self) -> List[AgentOutput]:
		"""Get all model outputs"""
		return [h.model_output for h in self.history if h.model_output]

	def model_actions(self) -> List[Dict[str, Any]]:
		"""Get all actions with their parameters"""
		outputs = []
		for h in self.history:
			if h.model_output:
				outputs.extend(
					action.model_dump(exclude_none=True) 
					for action in h.model_output.action
				)
		return outputs

	def action_results(self) -> List[ActionResult]:
		"""Get all action results"""
		return [r for h in self.history for r in h.result if r]

	def extracted_content(self) -> List[str]:
		"""Get all extracted content"""
		return [
			r.extracted_content 
			for h in self.history 
			for r in h.result 
			if r.extracted_content
		]

	def model_actions_filtered(self, include: List[str]) -> List[Dict[str, Any]]:
		"""Get filtered model actions matching specified names"""
		outputs = self.model_actions()
		return [
			o for o in outputs 
			if list(o.keys())[0] in include
		]


class AgentError:
	"""Error handling for agent operations"""

	VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
	RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
	NO_VALID_ACTION = 'No valid action found'

	@staticmethod
	def format_error(error: Exception, include_trace: bool = False) -> str:
		"""Format error message with optional stack trace"""
		if isinstance(error, ValidationError):
			return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
		if isinstance(error, RateLimitError):
			return AgentError.RATE_LIMIT_ERROR
		if include_trace:
			return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
		return str(error)
