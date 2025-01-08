from typing import Literal, Optional

from pydantic import BaseModel


class ActionResult(BaseModel):
	"""Result of executing an action."""
	is_done: bool = False
	extracted_content: Optional[str] = None
	error: Optional[str] = None
	include_in_memory: bool = False


# Action Input Models
class SearchGoogleAction(BaseModel):
	query: str


class GoToUrlAction(BaseModel):
	url: str


class ClickElementAction(BaseModel):
	index: int
	xpath: Optional[str] = None


class InputTextAction(BaseModel):
	index: int
	text: str
	xpath: Optional[str] = None


class DoneAction(BaseModel):
	text: str


class SwitchTabAction(BaseModel):
	page_id: int


class OpenTabAction(BaseModel):
	url: str


class ExtractPageContentAction(BaseModel):
	value: Literal['text', 'markdown', 'html'] = 'text'


class ScrollAction(BaseModel):
	amount: Optional[int] = None  # The number of pixels to scroll. If None, scroll down/up one page


class SendKeysAction(BaseModel):
	keys: str
