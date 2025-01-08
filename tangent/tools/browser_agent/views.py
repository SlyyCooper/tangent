from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, List

from openai import RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from tangent.wrappers.browser.views import BrowserStateHistory
from tangent.wrappers.controller.registry.views import ActionModel
from tangent.wrappers.dom.history_tree_processor.service import (
	DOMElementNode,
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from tangent.wrappers.dom.views import SelectorMap
from tangent.types import Structured_Result


@dataclass
class AgentStepInfo:
	step_number: int
	max_steps: int


class ActionResult(BaseModel):
	"""Result of executing an action"""

	is_done: Optional[bool] = False
	extracted_content: Optional[str] = None
	error: Optional[str] = None
	include_in_memory: bool = False  # whether to include in past messages as context or not


class AgentBrain(BaseModel):
	"""Current state of the agent"""

	evaluation: str
	memory: str
	next_goal: str


class AgentOutput(Structured_Result):
	"""Output model for browser agent actions that extends Tangent's Structured_Result.
	
	This class provides a structured way to represent browser agent outputs while maintaining
	compatibility with Tangent's output format. It includes both the agent's cognitive state
	(evaluation, memory, goals) and its planned actions.

	Attributes:
		result_overview (str): A human-readable summary of the agent's current state and next action
		extracted_data (dict): Structured data including evaluation, memory, goals, and planned actions
	"""
	model_config = ConfigDict(arbitrary_types_allowed=True)

	@staticmethod
	def type_with_custom_actions(custom_actions: Type[ActionModel]) -> Type['AgentOutput']:
		"""Creates a new AgentOutput type with custom action types.
		
		This method allows the AgentOutput to be extended with domain-specific actions
		while maintaining the structured output format required by Tangent.
		
		Args:
			custom_actions: The ActionModel type containing domain-specific actions
			
		Returns:
			A new AgentOutput type that includes the custom actions
		"""
		return create_model(
			'AgentOutput',
			__base__=AgentOutput,
			actions=(List[custom_actions], Field(...)),
			__module__=AgentOutput.__module__,
		)

	@classmethod
	def from_brain_and_actions(
		cls,
		brain: AgentBrain,
		actions: List[ActionModel]
	) -> 'AgentOutput':
		"""Creates an AgentOutput instance from brain state and actions.
		
		This method formats the agent's cognitive state and planned actions into
		Tangent's structured output format, making it compatible with both the
		browser agent and Tangent's expectations.
		
		Args:
			brain: The agent's current cognitive state
			actions: List of planned actions to execute
			
		Returns:
			An AgentOutput instance containing both the brain state and actions
			in Tangent's structured format
		"""
		return cls(
			result_overview=f"{brain.evaluation} - Next: {brain.next_goal}",
			extracted_data={
				"evaluation": brain.evaluation,
				"memory": brain.memory,
				"next_goal": brain.next_goal,
				"actions": [action.model_dump(exclude_none=True) for action in actions]
			}
		)


class AgentHistory(BaseModel):
	"""History item for agent actions"""

	model_output: AgentOutput | None
	result: list[ActionResult]
	state: BrowserStateHistory

	model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

	@staticmethod
	def get_interacted_element(
		model_output: AgentOutput, selector_map: SelectorMap
	) -> list[DOMHistoryElement | None]:
		elements = []
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
				action.model_dump(exclude_none=True) for action in self.model_output.action
			]
			model_output_dump = {
				'current_state': self.model_output.current_state.model_dump(),
				'action': action_dump,  # This preserves the actual action data
			}

		return {
			'model_output': model_output_dump,
			'result': [r.model_dump(exclude_none=True) for r in self.result],
			'state': self.state.to_dict(),
		}


class AgentHistoryList(BaseModel):
	"""List of agent history items"""

	history: list[AgentHistory]

	def __str__(self) -> str:
		"""Representation of the AgentHistoryList object"""
		return f'AgentHistoryList(all_results={self.action_results()}, all_model_outputs={self.model_actions()})'

	def __repr__(self) -> str:
		"""Representation of the AgentHistoryList object"""
		return self.__str__()

	def save_to_file(self, filepath: str | Path) -> None:
		"""Save history to JSON file with proper serialization"""
		try:
			Path(filepath).parent.mkdir(parents=True, exist_ok=True)
			data = self.model_dump()
			with open(filepath, 'w', encoding='utf-8') as f:
				json.dump(data, f, indent=2)
		except Exception as e:
			raise e

	def model_dump(self, **kwargs) -> Dict[str, Any]:
		"""Custom serialization that properly uses AgentHistory's model_dump"""
		return {
			'history': [h.model_dump(**kwargs) for h in self.history],
		}

	@classmethod
	def load_from_file(
		cls, filepath: str | Path, output_model: Type[AgentOutput]
	) -> 'AgentHistoryList':
		"""Load history from JSON file"""
		with open(filepath, 'r', encoding='utf-8') as f:
			data = json.load(f)
		# loop through history and validate output_model actions to enrich with custom actions
		for h in data['history']:
			if h['model_output']:
				if isinstance(h['model_output'], dict):
					h['model_output'] = output_model.model_validate(h['model_output'])
				else:
					h['model_output'] = None
			if 'interacted_element' not in h['state']:
				h['state']['interacted_element'] = None
		history = cls.model_validate(data)
		return history

	def last_action(self) -> None | dict:
		"""Last action in history"""
		if self.history and self.history[-1].model_output:
			return self.history[-1].model_output.action[-1].model_dump(exclude_none=True)
		return None

	def errors(self) -> list[str]:
		"""Get all errors from history"""
		errors = []
		for h in self.history:
			errors.extend([r.error for r in h.result if r.error])
		return errors

	def final_result(self) -> None | str:
		"""Final result from history"""
		if self.history and self.history[-1].result[-1].extracted_content:
			return self.history[-1].result[-1].extracted_content
		return None

	def is_done(self) -> bool:
		"""Check if the agent is done"""
		if (
			self.history
			and len(self.history[-1].result) > 0
			and self.history[-1].result[-1].is_done
		):
			return self.history[-1].result[-1].is_done
		return False

	def has_errors(self) -> bool:
		"""Check if the agent has any errors"""
		return len(self.errors()) > 0

	def urls(self) -> list[str]:
		"""Get all unique URLs from history"""
		return [h.state.url for h in self.history if h.state.url]

	def screenshots(self) -> list[str]:
		"""Get all screenshots from history"""
		return [h.state.screenshot for h in self.history if h.state.screenshot]

	def action_names(self) -> list[str]:
		"""Get all action names from history"""
		return [list(action.keys())[0] for action in self.model_actions()]

	def model_thoughts(self) -> list[AgentBrain]:
		"""Get all thoughts from history"""
		return [h.model_output.current_state for h in self.history if h.model_output]

	def model_outputs(self) -> list[AgentOutput]:
		"""Get all model outputs from history"""
		return [h.model_output for h in self.history if h.model_output]

	# get all actions with params
	def model_actions(self) -> list[dict]:
		"""Get all actions from history"""
		outputs = []

		for h in self.history:
			if h.model_output:
				for action in h.model_output.action:
					output = action.model_dump(exclude_none=True)
					outputs.append(output)
		return outputs

	def action_results(self) -> list[ActionResult]:
		"""Get all results from history"""
		results = []
		for h in self.history:
			results.extend([r for r in h.result if r])
		return results

	def extracted_content(self) -> list[str]:
		"""Get all extracted content from history"""
		content = []
		for h in self.history:
			content.extend([r.extracted_content for r in h.result if r.extracted_content])
		return content

	def model_actions_filtered(self, include: list[str] = []) -> list[dict]:
		"""Get all model actions from history as JSON"""
		outputs = self.model_actions()
		result = []
		for o in outputs:
			for i in include:
				if i == list(o.keys())[0]:
					result.append(o)
		return result


class AgentError:
	"""Container for agent error handling"""

	VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
	RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
	NO_VALID_ACTION = 'No valid action found'

	@staticmethod
	def format_error(error: Exception, include_trace: bool = False) -> str:
		"""Format error message based on error type and optionally include trace"""
		message = ''
		if isinstance(error, ValidationError):
			return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
		if isinstance(error, RateLimitError):
			return AgentError.RATE_LIMIT_ERROR
		if include_trace:
			return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
		return f'{str(error)}'