from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class DOMElementNode:
	"""Base class for DOM elements."""
	tag_name: str
	is_visible: bool
	parent: Optional['DOMElementNode']
	xpath: str
	attributes: Dict[str, str]
	children: List[Any]
	highlight_index: Optional[int] = None
	shadow_root: bool = False

	def get_all_text_till_next_clickable_element(self) -> str:
		"""Get all text content until the next clickable element."""
		text = []
		for child in self.children:
			if isinstance(child, str):
				text.append(child)
			elif isinstance(child, DOMElementNode):
				if child.highlight_index is not None:
					break
				text.append(child.get_all_text_till_next_clickable_element())
		return ' '.join(text)

@dataclass
class HashedDomElement:
	"""
	Hash of the dom element to be used as a unique identifier
	"""

	branch_path_hash: str
	attributes_hash: str
	# text_hash: str


@dataclass
class DOMHistoryElement:
	"""
	Represents a historical DOM element state
	"""

	tag_name: str
	xpath: str
	highlight_index: Optional[int]
	entire_parent_branch_path: list[str]
	attributes: dict[str, str]
	shadow_root: bool = False

	def to_dict(self) -> dict:
		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'highlight_index': self.highlight_index,
			'entire_parent_branch_path': self.entire_parent_branch_path,
			'attributes': self.attributes,
			'shadow_root': self.shadow_root,
		}
